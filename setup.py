from setuptools import setup
# To use a consistent encoding
from codecs import open
from os import path

# trying to import the required torch package
try:
    import torch
except ImportError:
    raise Exception('torchrsvd requires PyTorch to be installed. aborting')

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Proceed to setup
setup(
    name='torchrsvd',
    version='1.0',
    description='An implementation of (fast) randomized SVD for pytorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Antoine Liutkus',
    author_email='antoine.liutkus@inria.fr',
    packages=['torchrsvd'],
    keywords='svd torch',
    install_requires=[],
    )
