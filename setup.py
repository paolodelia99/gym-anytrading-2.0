from setuptools import setup, find_packages

setup(
    name='gym_anytrading2',
    version='0.0.1',
    packages=find_packages(),

    author='Paolo D\'Elia',
    author_email='paolo.delia99@gmail.com',

    install_requires=[
        'gym>=0.12.5',
        'numpy>=1.16.4',
        'pandas>=0.24.2',
        'matplotlib>=3.1.1'
    ],

    package_data={
        'gym_anytrading2': ['datasets/data/*']
    }
)
