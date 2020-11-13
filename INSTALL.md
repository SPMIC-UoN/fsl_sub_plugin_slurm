# Installation

## Requirements

fsl\_sub\_plugin\_slurm requires Python >=3.5 and fsl\_sub >=2.5.0

## Installation within FSL

FSL ships with fsl_sub pre-installed but lacking any grid backends. To install this backend use the fsl_sub_plugin helper script:

> $FSLDIR/bin/fsl_sub_plugin -i fsl_sub_plugin_slurm

## Installation outside FSL

### Conda

If you are using Conda then you can install the plugin with the following (note this will automatically install fsl\_sub if required):

> conda install -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/channel fsl_sub_plugin_slurm

### Virtualenv

If you are using a virtual env, make sure the environment containing fsl\_sub is active and then use:

> pip install git+ssh://git@git.fmrib.ox.ac.uk/fsl/fsl_sub_plugin_slurm.git
