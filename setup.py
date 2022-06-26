from setuptools import setup, find_packages

setup(
    name='gym_anytrading2',
    version='0.0.1',
    packages=find_packages(),

    author='Paolo D\'Elia',
    author_email='paolo.delia99@gmail.com',

    install_requires=[
        'gym>=0.21.0',
        'numpy>=1.22.3',
        'pandas>=1.4.1',
        'matplotlib>=3.2.1'
    ],

    package_data={
        'gym_anytrading2': ['datasets/data/*']
    }
)
