# fsl_sub_plugin_slurm

Job submission to SLURM variant cluster queues.
_Copyright 2018, University of Oxford (Duncan Mortimer)_

## Introduction

fsl_sub provides a consistent interface to various cluster backends, with a fall back to running tasks locally where no cluster is available.
This fsl_sub plugin provides support for submitting tasks to SLURM clusters.

## Requirements

fsl_sub_plugin_slurm requires Python >=3.5 and fsl_sub >=2.1.0

## Installation

Where fsl_sub is to be used outside of the FSL distribution it is recommended that it is installed within a Conda or virtual environment.

### Installation within FSL

FSL ships with fsl_sub pre-installed but lacking any grid backends. To install this backend use the fsl_sub_plugin helper script:

$FSLDIR/etc/fslconf/fsl_sub_plugin -i fsl_sub_plugin_sge

### Installing plugins

If you only need to run programs locally, fsl_sub ships with a local job plugin, but if you wish to target a grid backend you need to install an appropriate plugin. To install this plugin, ensure your environment is activated and then install the plugin with:

#### virtualenv

> pip install git+ssh://git@git.fmrib.ox.ac.uk/fsl/fsl_sub_plugin_sge.git

#### conda

> conda install fsl_sub_plugin_sge

### Configuration

A configuration file in YAML format is required to describe your cluster environment, an example configuration can be generated with:

> fsl_sub_config sge > fsl_sub.yml

This configuration file can be copied to _fsldir_/etc/fslconf (calling it fsl_sub.yml), or put in your home folder calling it .fsl_sub.yml. A copy in your home folder will override the file in _fsldir_/etc/fslconf.

Finally, the environment variable FSLSUB_CONF can be set to point at the location of this configuration file, this will override all other files.

#### Plugin Configuration Sections

In the _method\_opts_ section, a definition for the method _slurm_ needs to be edited/added.

##### Method options

* queues: True/False - does this method use queues/partitions (should be True)
* large_job_split_pe: - not used for slurm so leave as the default
* copy_environment: True/False - whether to replicate the environment variables in the shell that called fsl_sub into the job's shell
* script_conf: True/False - whether _--usesscript_ option to fsl_sub is available via this method. This option allows you to define the slurm options as comments in a shell script and then provide this to the cluster for running. Should be set to True.
* mail_support: True/False - whether the slurm installation is configured to send email on job events.
* mail_modes: If the grid has email notifications turned on, this option configures the submission options for different verbosity levels, 'b' = job start, 'e' = job end, 'a' = job abort, 'f' = all events, 'n' = no mail. Each event type should then have a list of submission mail arguments that will be applied to the submitted job. Typically, these should not be edited.
* mail_mode: Which of the above mail_modes to use by default
* map_ram: True/False - if a job requests more RAM than is available in any one queue whether fsl_sub should request sufficient cpus to achieve this memory request, e.g. if your maximum slot size is 16GB and you request 64GB if this option is on then fsl_sub will request four cpus. As a side-effect your job will now be free to use four threads.
* thread_ram_divide: True/False - If you have requested a multi-threaded job, does Slurm expect you to specify the appropriate fraction of the total memory required (True) or the total memory of the task (False). For Slurm this should normally be left at False.
* notify_ram_usage: True/False - Whether to notify Slurm of the RAM you have requested. Advising the grid software of your RAM requirements can help with scheduling or may be used for special features (such as RAM disks). Use this to control whether fsl_sub passes on your RAM request to the scheduler.
* array_holds: True/False - does grid software support array holds, e.g. sub-task 1 waits for parent sub-task 1. True for Slurm > 16.05
* array_limit - True/False - does grid software support limiting number of sub-tasks running concurrently. True Slurm.
* job_resources - True/False - does the grid software accept additional job resource specifications? True for Slurm.
* projects - True/False - does the grid software support accounts for auditing/charging purposes? Typically true for Slurm, but implementation dependent.

##### Coprocessor Configuration

This section defines what coprocessors are available in your cluster. This would typically be used to inform fsl_sub of CUDA resources. By default it is commented out, but if you have CUDA capable hardware present on your cluster then uncomment this section to enable the coprocessor submission options.

For each coprocessor hardware type you need a sub-section with an identifier than will be used to request this type of coprocessor, e.g. _cuda_. Then in the sub-section there are the following options:

* resource: GRES resource that, when requested, selects machines with the hardware present. For CUDA/GPU hardware this is normally _gpu_. You can get the name of this for each partition with the command (resource appears after the comma):
  > sinfo -a -o "%P,%G"
* classes: True/False - does the cluster contain (and differentiate between) multiple classes of coprocessor, e.g. Kepler, Pascal and Volta Tesla devices.
* include_more_capable: True/False - Whether to limit to the exact class specifed (False) or allow running on higher capability devices (True) by default. The user can override this setting.
* uses_modules: True/False - Is the coprocessor configured using a shell module?
* module_parent: If you use shell modules, what is the name of the parent module? e.g. _cuda_ if you have a module folder _cuda_ with module files within for the various CUDA versions.
* slurm_constraints: True/False - Does your Slurm install use _constraints_ to specify GPU types? Defaults to True
* class_types: This contains the definition of the GPU classes...
  * class selector: This is the letter (or word) that is used to select this class of co-processor from the fsl_sub commandline. In the case of CUDA GPUs it should be the letter designating the GPU family, e.g. K, P or V.
    * resource: This is the constraint name that will be used to select this GPU family, you can get the available options with the command (constraint appears after the comma)
      > sinfo -a -o "%P,%f"
    * doc: The description that appears in the fsl_sub help text about this device
    * capability: An integer defining the feature set of the device, your most basic device should be given the value 1 and more capable devices higher values, e.g. GTX = 1, Kelper = 2, Pascal = 3, Volta = 4.
* default_class: The _class selector_ for the class to assign jobs to where a class has not been specified in the fsl_sub call.

##### Queue Definitions

Slurm refers to queues as partitions. To determine the information necessary to configure the partition definitions you can use the following tools:

Each queue is defined in a YAML dictionary, keyed on the name of the partition. To get a list of all available partitions use:

> sinfo -s

The partition names are given in the first column.

Then the details for a queue can be obtained with:

> sinfo -p _partitionname_ -Nel

This will return details for every node within that partition.

> sinfo -p _partitionname_ --long

This will give a summary of the nodes available on the queue and provides job limit information.

The settings for the queue can be found in the following partition properties:

* time: _TIMELIMIT_ reports in days-hours:minutes:seconds, this needs converting to seconds (from the _--long_ output)
* slot_size: This is the maximum permitted memory per thread (cpu/core) converted to units specified in the main configuration for fsl_sub. This is usually reported in MB, so for example 63000 should be configured as 63 if you have configured fsl_sub to expect GB. It is equal to the total memory (_MEMORY_) of a node divided by the number of cores (_CPUS_). Where a partition has nodes of differing memory size specify the largest figure and ensure that you have _notify\_ram\_usage_ turned on so that Slurm can auto-select the node based on your RAM request.
* max_slots: _CPU_ contains the number of CPUs (threads) available on each node. Set this option to the maximum number reported.
* group: An integer that allows grouping similar queues together, all queues in the same group will be candidates for a job that matches their capabilities
* priority: An integer that specifies an order for queues within a group, smaller = higher priority.
* map_ram: Whether to automatically submit large jobs as multiple threads to achieve the memory requested.

The final property _max\_size_ is the maximum requestable RAM by a job. This should be set to the smaller of (a) the maximum size for the _MEMORY_ property for hosts in this partition, (b) the number after the _-_ in the _JOB\_SIZE_ field in the _--long_ output (if this is _infinite_ then use the _MEMORY_ property).

If the queue is a coprocessor queue then you need to provide the following additional properties in a dictionary called _copros_, with sub-dictionaries keys on the coprocessor name this queue provides.
The information necessary to configure this can be obtained with this command:

> sinfo -a -o "%P,%G,%f"

This returns the partition name, resource and constraint for each unique combination of the three options. Using this information complete these configuration options:

* max_quantity: an integer representing the maximum number of this coprocessor type available on a single compute node. For each partition, this is the number after the ":" in the resource field. Where there are multiple definitions for a partition with the same constraint (e.g. 2x K80 on some nodes, 4x K80 on others) use the higher of the two quantities.
* classes: a list of coprocessor classes (as defined in the coprocessor configuration section) that this queue has hardware for. This is the constraint field, list all that appear in lines associated with this partition.

e.g.

  copros:
    cuda:
       max_quantity: 2
       classes:
         - K
         - P

## Building

Prepare the source distribution

> python setup.py sdist

To build a wheel you need to install wheel into your Python build environment

> pip install wheel

fsl_sub is only compatible with python 3 so you will be building a Pure Python Wheel

> python setup.py bdist_wheel

To build a conda package:

> cd condaRecipies
> conda-build ..