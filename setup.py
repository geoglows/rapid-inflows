from setuptools import setup, find_packages
import re

NAME = 'basininflow'
DESCRIPTION = 'Calculates time series of runoff for basins from gridded runoff LSM data'
URL = 'https://github.com/geoglows/basininflow'
AUTHOR = 'Riley Hales PhD'
REQUIRES_PYTHON = '>=3'
LICENSE = 'CC BY 4.0'

with open(f'./{NAME}/__init__.py') as f:
    version_pattern = r'__version__ = [\'"](\d+\.\d+\.\d+)[\'"]'
    VERSION = re.search(version_pattern, f.read()).group(1)

with open('./requirements.txt') as f:
    INSTALL_REQUIRES = f.read().splitlines()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=INSTALL_REQUIRES,
    include_package_data=False,
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Hydrology',
    ],
    entrypoints={
        'console_scripts': [
            'basininflow = basininflow.cli:main',
        ],
    },
)
