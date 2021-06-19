"""
Setup for Hierarchical VQ-VAE
"""

from setuptools import setup, find_packages

setup(
    name='Hierarchical VQ-VAE',
    version='0.0.0',
    packages=['HVQVAE'],
    package_data=[('HVQVAE', 'HVQVAE/hvqvae_weights/top_encoder.h5')],
    install_requires=[
       'numpy',
       'tensorflow',
       'matplotlib',
       'tensorflow-probability'
       
    ]
)
