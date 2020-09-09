#!/usr/bin/env python
from setuptools import setup, find_packages
from .fsl_sub_plugin_sge import plugin_version

setup(
    name='fsl_sub_plugin_slurm',
    version=plugin_version(),
    description='FSL Cluster Submission Plugin for Slurm',
    author='Duncan Mortimer',
    author_email='duncan.mortimer@ndcn.ox.ac.uk',
    url='https://git.fmrib.ox.ac.uk/fsl/fsl_sub_plugin_slurm',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Natural Language :: English',
        'Environment :: Console',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
    ],
    keywords='FSL fsl Neuroimaging neuroimaging cluster'
             ' grid slurm grid engine',
    project_urls={
        'Documentation': 'https://fsl.fmrib.ox.ac.uk/fsl/fslwiki',
        'Source': 'https://git.fmrib.ox.ac.uk/fsl/fsl_sub_plugin_slurm'
    },
    packages=find_packages(),
    license='FSL License',
    install_requires=['fsl_sub>=2.3.0', ],
    python_requires='~=3.5',
    package_data={
        'fsl_sub_plugin_slurm': ['fsl_sub_slurm.yml'],
    },
    include_package_data=True,
)
