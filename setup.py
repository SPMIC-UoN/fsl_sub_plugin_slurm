#!/usr/bin/env python
from setuptools import setup, find_packages

with open('fsl_sub_plugin_slurm/version.py', mode='r') as vf:
    vl = vf.read().strip()

PLUGIN_VERSION = vl.split(' = ')[1].strip("'")

setup(
    name='fsl_sub_plugin_slurm',
    version=PLUGIN_VERSION,
    description='FSL Cluster Submission Plugin for Slurm',
    author='Duncan Mortimer',
    author_email='duncan.mortimer@ndcn.ox.ac.uk',
    url='https://git.fmrib.ox.ac.uk/fsl/fsl_sub_plugin_slurm',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Natural Language :: English',
        'Environment :: Console',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
    ],
    keywords='FSL fsl Neuroimaging neuroimaging cluster'
             ' grid slurm',
    project_urls={
        'Documentation': 'https://fsl.fmrib.ox.ac.uk/fsl/fslwiki',
        'Source': 'https://git.fmrib.ox.ac.uk/fsl/fsl_sub_plugin_slurm'
    },
    packages=find_packages(),
    license='FSL License',
    install_requires=['fsl_sub>=2.5.6', 'ruamel.yaml>=0.16.7', ],
    setup_requires=['ruamel.yaml', ],
    python_requires='~=3.6',
    package_data={
        'fsl_sub_plugin_slurm': ['fsl_sub_slurm.yml', 'README.md', 'CHANGES.md', 'BUILD.md', 'INSTALL.md', ],
    },
    include_package_data=True,
)
