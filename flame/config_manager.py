# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
from collections import defaultdict
from typing import Tuple

import torch

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from flame.logging import logger

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def string_list(raw_arg):
    """Comma-separated string list argument."""
    return [s.strip() for s in raw_arg.split(",") if s.strip()]


def check_string_list_argument(args_dict: dict[str, any], fullargname: str):
    section, name = fullargname.split(".")
    # Split string list which are still raw strings.
    if (
        section in args_dict
        and name in args_dict[section]
        and isinstance(args_dict[section][name], str)
    ):
        sec = args_dict[section]
        sec[name] = string_list(sec[name])


class JobConfig:
    """
    A helper class to manage the train configuration.
    Semantics:
    - Default config is loaded from a toml file. If no toml file is provided,
    then the default config is loaded from argparse defaults.
    - if toml file has missing keys, they are filled with argparse defaults.
    - if additional explicit cmd args are provided in addition to the toml
    file, they will override the toml config and the argparse defaults

    precedence order: cmdline > toml > argparse default

    Arg parsing semantics:

    Each argument starts with <prefix>_ which is the section name in the toml file
    followed by name of the option in the toml file. For ex,
    model.name translates to:
        [model]
        name
    in the toml file
    """

    def __init__(self):
        self.args_dict = None
        # main parser
        self.parser = argparse.ArgumentParser(description="flame arg parser.")

        self.parser.add_argument(
            "--job.config_file",
            type=str,
            default=None,
            help="Job config file",
        )

        # job level configs
        self.parser.add_argument(
            "--job.dump_folder",
            type=str,
            default="./flame/outputs",
            help="Folder to dump job outputs",
        )
        self.parser.add_argument(
            "--job.description",
            type=str,
            default="default job",
            help="Description of the job",
        )
        self.parser.add_argument(
            "--job.use_for_integration_test",
            action="store_true",
            help="Add this config to the integration test suite",
        )
        self.parser.add_argument(
            "--job.print_args",
            action="store_true",
            help="Print the args to terminal",
        )

        # model configs
        self.parser.add_argument(
            "--model.name",
            type=str,
            default="fla",
            help="Which model to train",
        )
        self.parser.add_argument(
            "--model.config",
            type=str,
            default="fla-hub/transformer-1.3B-100B",
            help="Path to the model config",
        )
        self.parser.add_argument(
            "--model.tokenizer_path",
            type=str,
            default="fla-hub/transformer-1.3B-100B",
            help="Tokenizer path",
        )
        self.parser.add_argument(
            "--model.converters",
            type=string_list,
            nargs="+",
            default=[],
            help="""
                Comma separated list of converters to apply to the model.
                For instance, the `float8` converter swaps `torch.nn.Linear`
                with `Float8Linear`. This feature requires you to install 'torchao'
                which can be found here: https://github.com/pytorch/ao
            """,
        )
        self.parser.add_argument(
            "--model.print_after_conversion",
            action="store_true",
            help="""
            If true, model definition will be printed to stdout after all model
            converters have been applied.
            """,
        )

        # profiling configs (torch.profiler)
        self.parser.add_argument(
            "--profiling.enable_profiling",
            action="store_true",
            help="Whether to enable pytorch profiler",
        )
        self.parser.add_argument(
            "--profiling.save_traces_folder",
            type=str,
            default="profile_traces",
            help="Trace files location",
        )
        self.parser.add_argument(
            "--profiling.profile_freq",
            type=int,
            default=10,
            help="How often to collect profiler traces, in iterations",
        )

        # optimizer configs
        self.parser.add_argument(
            "--optimizer.name", type=str, default="AdamW", help="Optimizer to use"
        )
        self.parser.add_argument(
            "--optimizer.eps",
            type=float,
            default=1e-8,
            help="Epsilon value for the optimizer.",
        )
        self.parser.add_argument(
            "--optimizer.lr", type=float, default=8e-4, help="Learning rate to use"
        )
        self.parser.add_argument(
            "--optimizer.beta1", type=float, default=0.9,
            help="Exponential moving average hyperparameters to use"
        )
        self.parser.add_argument(
            "--optimizer.beta2", type=float, default=0.95,
            help="Exponential moving average hyperparameters to use"
        )
        self.parser.add_argument(
            "--optimizer.weight_decay", type=float, default=0.1,
            help="Weight decay to use"
        )
        self.parser.add_argument(
            "--optimizer.implementation",
            type=str,
            default="fused",
            choices=["for-loop", "foreach", "fused"],
            help="""
            Specify which optimizer implementation to use:
            - 'fused': Use fused implementation (CUDA only) for best performance.
            - 'foreach': Use some horizontal fusion of tensors for better performance.
            - 'for-loop': Use the default implementation for the optimizer (slowest).
            - more info: https://pytorch.org/docs/stable/optim.html
            """,
        )

        # lr scheduler configs
        self.parser.add_argument(
            "--lr_scheduler.warmup_steps",
            type=int,
            default=200,
            help="Steps for lr scheduler warmup, normally 1/5 of --training.steps",
        )
        self.parser.add_argument(
            "--lr_scheduler.decay_ratio",
            type=float,
            default=None,
            help="""
            Controls the proportion of the training steps allocated to the learning rate decay phase.

            If `None`, the learning rate will begin decaying immediately after the warmup period.
            Otherwise, the learning rate will remain stable after the warmup period and
            only start decaying during the last `decay_ratio` portion of the total training steps.

            This is known as the Warmup-Stable-Decay (WSD) schedule, as described in https://arxiv.org/abs/2404.06395.
            """,
        )
        self.parser.add_argument(
            "--lr_scheduler.decay_type",
            type=str,
            default="linear",
            choices=["linear", "sqrt", "cosine"],
            help="""
            Learning rate decay type to use during training:
            - 'linear': linearly decays learning rate from initial to final value
            - 'sqrt': decays learning rate following a 1 minus square root curve
            - 'cosine': smoothly decays learning rate following a cosine curve
            """,
        )
        self.parser.add_argument(
            "--lr_scheduler.lr_min",
            type=float,
            default=0.0,
            help="""
            Min lr ratio for lr scheduler.

            If provided, the range of decay factor is scaled from 1 to `lr_min`
            to ensure the learning rate does not drop below `optimizer.lr * lr_scheduler.lr_min`.
            """,
        )

        # training configs
        self.parser.add_argument(
            "--training.batch_size", type=int, default=8, help="Batch size"
        )
        self.parser.add_argument(
            "--training.seq_len", type=int, default=2048, help="Sequence length"
        )
        self.parser.add_argument(
            "--training.context_len",
            type=int,
            default=2048,
            help="Max length allowed for each sequence",
        )
        self.parser.add_argument(
            "--training.varlen",
            action="store_true",
            help="Whether to take sequences of variable length as input",
        )
        self.parser.add_argument(
            "--training.gradient_accumulation_steps",
            type=int,
            default=1,
            help="Number of steps to accumulate gradients before updating parameters",
        )
        self.parser.add_argument(
            "--training.steps",
            type=int,
            default=10000,
            help="How many train steps to run",
        )
        self.parser.add_argument(
            "--training.max_norm",
            type=float,
            default=1.0,
            help="Max norm for gradient clipping",
        )
        self.parser.add_argument(
            "--training.skip_nan_inf",
            action="store_true",
            help="Skip batch updates when NaN or INF gradients are encountered during training",
        )
        self.parser.add_argument(
            "--training.dataset",
            default="HuggingFaceFW/fineweb-edu",
            help="Dataset to use, with comma separated values",
        )
        self.parser.add_argument(
            "--training.dataset_name",
            default=None,
            help="The name of the dataset config, with comma separated values if provided",
        )
        self.parser.add_argument(
            "--training.dataset_split",
            default=None,
            help="Dataset split to use, with comma separated values if provided",
        )
        self.parser.add_argument(
            "--training.data_dir",
            default=None,
            help="Data dirs to use, with comma separated values if provided",
        )
        self.parser.add_argument(
            "--training.data_files",
            default=None,
            help="Data files to use, with comma separated values if provided",
        )
        self.parser.add_argument(
            "--training.data_probs",
            default=None,
            help="Data sampling probabilities, with comma separated values if provided",
        )
        self.parser.add_argument(
            "--training.streaming",
            action="store_true",
            help="Whether to load dataset in streaming mode, used for huge dataset",
        )
        self.parser.add_argument(
            "--training.num_workers",
            type=int,
            default=32,
            help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
        )
        self.parser.add_argument(
            "--training.prefetch_factor",
            type=int,
            default=2,
            help="Number of batches loaded in advance by each worker."
            "2 means there will be a total of 2 * num_workers batches prefetched across all workers.",
        )
        self.parser.add_argument(
            "--training.pin_memory",
            action="store_true",
            help="Whether to pin memory for data loading",
        )
        self.parser.add_argument(
            "--training.persistent_workers",
            action="store_true",
            help="Whether to use persistent workers for data loading",
        )
        self.parser.add_argument(
            "--training.mixed_precision_param",
            type=str,
            default="bfloat16",
            choices=["bfloat16", "float16", "float32"],
            help="dtype for mixed-precision training (passed to Accelerator as mixed_precision).",
        )
        self.parser.add_argument(
            "--training.compile",
            action="store_true",
            help="Whether to compile the model",
        )
        self.parser.add_argument(
            "--training.gc_freq",
            type=int,
            default=50,
            help="Python garbage control scheduling interval, in steps",
        )
        self.parser.add_argument(
            "--training.seed",
            type=int,
            default=42,
            help="Choose the base RNG seed used for training",
        )
        self.parser.add_argument(
            "--training.deterministic",
            action="store_true",
            help="Use deterministic algorithms wherever possible, may be slower",
        )
        # metrics configs
        self.parser.add_argument(
            "--metrics.log_freq",
            type=int,
            default=10,
            help="How often to log metrics to TensorBoard, in iterations",
        )
        self.parser.add_argument(
            "--metrics.enable_tensorboard",
            action="store_true",
            help="Whether to log metrics to TensorBoard",
        )
        self.parser.add_argument(
            "--metrics.disable_color_printing",
            action="store_true",
            help="Whether to disable color printing in logs",
        )
        self.parser.add_argument(
            "--metrics.save_tb_folder",
            type=str,
            default="tb",
            help="Folder to dump TensorBoard states",
        )
        self.parser.add_argument(
            "--metrics.save_for_all_ranks",
            action="store_true",
            default=False,
            help="""
                Whether to save TensorBoard/Wandb metrics only for rank 0 or for all ranks.
                When this option is False and pipeline_parallel_degree is > 1, the metrics
                component uses the 0th rank of the last stage pipeline group, which is the
                only stage that computes loss metrics.
            """,
        )
        self.parser.add_argument(
            "--metrics.enable_wandb",
            action="store_true",
            help="Whether to log metrics to Weights & Biases",
        )

        self.parser.add_argument(
            "--experimental.custom_model_path",
            type=str,
            default="",
            help="""
                Path (filesystem or dotted import) to a custom model module that is not
                built into flame.  e.g. 'my_models/model_x' or 'some_package.model_x'.
            """,
        )
        self.parser.add_argument(
            "--checkpoint.enable_checkpoint",
            action="store_true",
            help="Whether to enable checkpoint",
        )
        self.parser.add_argument(
            "--checkpoint.folder",
            type=str,
            default="checkpoint",
            help="The folder to store the checkpoints (relative to --job.dump_folder).",
        )
        self.parser.add_argument(
            "--checkpoint.initial_load_path", type=str, default=None,
            help="""
                Path to an initial checkpoint to load (useful for resuming with a different
                output path or for warm-starting from a pre-trained model).
                If the current checkpoint folder is non-empty, this option is ignored.
            """
        )
        self.parser.add_argument(
            "--checkpoint.interval",
            type=int,
            default=500,
            help="Checkpointing interval in steps.",
        )
        self.parser.add_argument(
            "--checkpoint.last_save_model_weights_only",
            action="store_true",
            help="When True, only model weights are saved at the end of training.",
        )
        self.parser.add_argument(
            "--checkpoint.export_dtype",
            type=str,
            default="float32",
            choices=["float16", "bfloat16", "float32"],
            help="Cast model weights to this dtype when saving at the end of training.",
        )
        self.parser.add_argument(
            "--checkpoint.keep_latest_k",
            type=int,
            default=0,
            help="Keep only the latest k checkpoints; 0 means keep all.",
        )
        self.parser.add_argument(
            "--checkpoint.load_step",
            type=int,
            default=-1,
            help="Resume from the checkpoint at this step. -1 loads the latest checkpoint.",
        )
        self.parser.add_argument(
            "--checkpoint.exclude_from_loading",
            type=string_list,
            nargs="*",
            default=[],
            help="Comma-separated list of checkpoint keys to skip when loading (e.g. 'optimizer,lr_scheduler').",
        )
        # activation checkpointing configs
        self.parser.add_argument(
            "--activation_checkpoint.mode",
            type=str,
            default="selective",
            help="Type of activation checkpointing to use ['none', 'full', 'selective']",
        )
        self.parser.add_argument(
            "--activation_checkpoint.selective_ac_option",
            type=str,
            default="2",  # 2 = checkpoint every other layer
            help="""
                Selective activation checkpointing options ['int', 'op'].
                'int' (e.g., 2) for every nth layer, or 'op' for op level ac.
            """,
        )

        self.parser.add_argument(
            "--activation_offload.mode",
            type=str,
            default="none",
            help="""
                if we are using activation offload or not. Options are ['none', 'full'].
            """,
        )

        # float8 training (requires torchao)
        self.parser.add_argument(
            "--float8.recipe_name",
            type=str,
            default=None,
            choices=["tensorwise", "rowwise", "rowwise_with_gw_hp"],
            help="If specified, apply float8 training using the given torchao recipe.",
        )

        # nvfp4 configs
        self.parser.add_argument(
            "--nvfp4.filter_fqns",
            type=string_list,
            nargs="+",
            default=[],
            help="""
                Comma-separated list of FQN substrings identifying modules to skip
                when applying NVFP4 quantization. For example, 'lm_head' will prevent
                the language model head from being converted to NVFP4Linear.
            """,
        )

        # bitsandbytes (bnb) configs
        self.parser.add_argument(
            "--bnb.quant_type",
            type=str,
            default="int8",
            choices=["int8", "fp4", "nf4"],
            help="""
                Quantization type for BitsAndBytes conversion.
                'int8': LLM.int8() 8-bit quantization (Linear8bitLt).
                'fp4':  FP4 4-bit quantization (LinearFP4).
                'nf4':  NF4 4-bit quantization from QLoRA (LinearNF4).
            """,
        )
        self.parser.add_argument(
            "--bnb.filter_fqns",
            type=string_list,
            nargs="+",
            default=[],
            help="""
                Comma-separated list of FQN substrings identifying modules to skip
                when applying BitsAndBytes quantization. For example, 'lm_head' will
                prevent the language model head from being converted.
            """,
        )

    def to_dict(self):
        return self.args_dict

    def parse_args(self, args_list: list = sys.argv[1:]):
        args, cmd_args = self.parse_args_from_command_line(args_list)
        config_file = getattr(args, "job.config_file", None)
        # build up a two level dict
        args_dict = self._args_to_two_level_dict(args)
        if config_file is not None:
            try:
                with open(config_file, "rb") as f:
                    for k, v in tomllib.load(f).items():
                        # to prevent overwrite of non-specified keys
                        args_dict[k] |= v
            except (FileNotFoundError, tomllib.TOMLDecodeError) as e:
                logger.exception(
                    f"Error while loading the configuration file: {config_file}"
                )
                logger.exception(f"Error details: {str(e)}")
                raise e

        # Checking string-list arguments are properly split into a list
        # if split-points came from 'args' (from cmd line) it would have already been parsed into a list by that parser
        string_list_argnames = self._get_string_list_argument_names()
        for n in string_list_argnames:
            check_string_list_argument(args_dict, n)

        # override args dict with cmd_args
        cmd_args_dict = self._args_to_two_level_dict(cmd_args)
        for section, section_args in cmd_args_dict.items():
            for k, v in section_args.items():
                args_dict[section][k] = v

        self.args_dict = args_dict

        for k, v in args_dict.items():
            class_type = type(k.title(), (), v)
            setattr(self, k, class_type())
        self._validate_config()

    def _args_to_two_level_dict(self, args: argparse.Namespace) -> defaultdict:
        args_dict = defaultdict(defaultdict)
        for k, v in vars(args).items():
            first_level_key, second_level_key = k.split(".", 1)
            args_dict[first_level_key][second_level_key] = v
        return args_dict

    def _validate_config(self) -> None:
        # TODO: Add more mandatory validations
        assert self.model.config
        assert self.model.tokenizer_path

    def _get_string_list_argument_names(self) -> list[str]:
        """Get the parser argument names of type `string_list`."""
        string_list_args = [
            v.dest for v in self.parser._actions if v.type is string_list
        ]
        return string_list_args

    def parse_args_from_command_line(
        self, args_list
    ) -> Tuple[argparse.Namespace, argparse.Namespace]:
        """
        Parse command line arguments and return the parsed args and the command line only args
        """
        args = self.parser.parse_args(args_list)
        string_list_argnames = set(self._get_string_list_argument_names())

        # aux parser to parse the command line only args, with no defaults from main parser
        aux_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        for arg, val in vars(args).items():
            if isinstance(val, bool):
                aux_parser.add_argument(
                    "--" + arg, action="store_true" if val else "store_false"
                )
            elif arg in string_list_argnames:
                # without this special case, type inference breaks here,
                # since the inferred type is just 'list' and it ends up flattening
                # e.g. from ["layers.0", "layers.1"] into ["l", "a", "y", "e", "r", "s", ".0", ...]
                aux_parser.add_argument("--" + arg, type=string_list)
            else:
                aux_parser.add_argument("--" + arg, type=type(val))

        cmd_args, _ = aux_parser.parse_known_args(args_list)

        return args, cmd_args
