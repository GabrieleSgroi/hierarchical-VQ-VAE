"""
Setup for Hierarchical VQ-VAE
"""

from setuptools import setup, find_packages

setup(
    name='Hierarchical VQ-VAE',
    version='0.0.0',
    packages=['HVQVAE'],
    include_package_data=True,
    install_requires=[
       'numpy',
       'tensorflow',
       'matplotlib',
       'tensorflow-probability',
       'progressbar2'
       
    ]
)
