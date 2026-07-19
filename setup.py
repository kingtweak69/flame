# -*- coding: utf-8 -*-

import ast
import os
import re
from pathlib import Path

from setuptools import find_packages, setup

with open('README.md') as f:
    long_description = f.read()


def get_package_version():
    with open(Path(os.path.dirname(os.path.abspath(__file__))) / 'flame' / '__init__.py') as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    return ast.literal_eval(version_match.group(1))


setup(
    name='flame',
    version=get_package_version(),
    description='A minimal training framework for scaling FLA models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Songlin Yang, Yu Zhang',
    author_email='yangsl66@mit.edu, yzhang.cs@outlook.com',
    url='https://github.com/fla-org/flame',
    packages=find_packages(),
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    python_requires='>=3.10',
    install_requires=[
        'flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention.git@v0.5.2',
        'torchtitan @ git+https://github.com/pytorch/torchtitan.git@0b44d4c',
        'torch',
        'torchdata',
        'transformers<5.0',
        'triton>=3.1.0',
        'datasets>=3.5.0',
        'einops',
        'ninja',
        'tyro',
        'wandb',
        'tiktoken',
        'tensorboard',
        'bitsandbytes',
        'nvidia-modelopt',
    ],
)
