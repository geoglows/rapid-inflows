from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    install_requires = f.readlines()

setup(
    name='basininflow',
    version='0.1.0',
    packages=find_packages(),
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'inflows = basininflow.cli:main',
        ],
    },
)
