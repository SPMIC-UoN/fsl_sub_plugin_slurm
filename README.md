# fsl_sub_plugin_slurm

Job submission to SLURM variant cluster queues.
_Copyright 2018-2020, University of Oxford (Duncan Mortimer)_

## Introduction

fsl\_sub provides a consistent interface to various cluster backends, with a fall back to running tasks locally where no cluster is available.
This fsl\_sub plugin provides support for submitting tasks to SLURM clusters.

For installation instructions please see INSTALL.md; for building packages see BUILD.md.

## Configuration

Use the command:

> fsl_sub_config slurm > fsl_sub.yml

to generate an example configuration, including queue definitions gleaned from the SLURM software - check these, paying attention to any warnings generated.

Use the fsl_sub.yml file as per the main fsl_sub documentation.

The configuration for the SLURM plugin is in the _method\_opts_ section, under the key _slurm_.

### Method options

| Key | Values (default/recommended in bold) | Description |
| ----|--------|-------------|
| queues | **True** | Does this method use queues/partitions (should be always be True) |
| memory\_in\_gb | True/**False** | Whether SLURM reports memory in GB - normally false |
| copy\_environment | **True**/False | Whether to replicate the environment variables in the shell that called fsl_sub into the job's shell
| script\_conf | **True**/False | Whether _--usesscript_ option to fsl_sub is available via this method. This option allows you to define the grid options as comments in a shell script and then provide this to the cluster for running. Should be set to True.
| mail\_support | True/**False** | Whether the grid installation is configured to send email on job events.
| mail\_modes | Dictionary of option lists | If the grid has email notifications turned on, this option configures the submission options for different verbosity levels, 'b' = job start, 'e' = job end, 'a' = job abort, 'f' = all events, 'n' = no mail. Each event type should then have a list of submission mail arguments that will be applied to the submitted job. Typically, these should not be edited.
| mail\_mode | b/e/a/f/**n** | Which of the above mail_modes to use by default
| notify\_ram\_usage | **True**/False | Whether to notify SLURM of the RAM you have requested. SLURM is typically configured to give jobs a small RAM allocation so you will invariably need this set to true.
| set\_time\_limit | True/**False** | Whether to notify SLURM of the expected *maximum* run-time of your job. This helps the scheduler fill in reserved slots (for e.g. parallel environment jobs), however, this time limit will be enforced, resulting in a job being killed if it is exceeded, even if this is less than the queue run-time limit. This can be disabled on a per-job basis by setting the environment variable FSLSUB_NOTIMELIMIT to '1' (or 'True').
| array\_holds | **True**/False | Enable support array holds, e.g. sub-task 1 waits for parent sub-task 1.
| array\_limit | **True**/False | Enable limiting number of concurrent array tasks.
| job\_resources | **True**/False | Enable additional job resource specification support.
| projects | **True**/False | Enable support for projects typically used auditing/charging purposes.
| preseve\_modules | **True**/False | Requires (and will enforce) use_jobscript. Whether to re-load shell modules on the compute node. Required if you have multiple CPU generations and per-generation optimised libraries configured with modules.
| add_module_paths | **[]**/ a list | List of file system paths to search for modules in addition to the system defined ones. Useful if you have your own shell modules directory but need to allow the compute node to auto-set it's MODULEPATH environment variable (e.g. to a architecture specific folder). Only used when preserve_modules is True.
| export\_vars | **[]**/List | List of environment variables that should transfered with the job to the compute node
| keep\_jobscript | True/**False** | Whether to preserve the generated wrapper in a file `jobid_wrapper.sh`. This file contains sufficient information to resubmit this job in the future.

### Coprocessor Configuration

This plugin is not capable of automatically determining all the necessary information to configure your co-processors but will advise of the information it can find and propose queue definitions for these GPU resources.

SLURM typically selects GPU resources with a GRES (Generic RESource) that defines the type and quantity of the co-processor. Where multiple classes of co-processor are available this might be selectable via the GRES or you may need to provide a _constraint_. If you would like to be able to support running on a class and all superior devices you need to be able to use constraints as GRES requests do not support logical combinations. The automatically generated configuration should include useful information about your GRES and constraints, but should you wish to obtain this information yourself use the commands:

* `sinfo -p <partition> -o %G` - This will list all the GRES defined on \<partition>.
* `sinfo -p <partition> -o %f` - This will list all features selectable by a `--constraint` as a comma-separated list.

Typically CUDA resources will be controlled using GRES or constraints with _gpu_ in the name, so look for these.

For each coprocessor hardware type you need a sub-section given an identifier than will be used to request this type of coprocessor. For CUDA processors this sub-section *must* be called 'cuda' to ensure that FSL tools can auto-detect and use CUDA hardware/queues.

| Key | Values (default/recommended in bold) | Description |
| ----|--------|-------------|
| resource| String | GRES that, when requested, selects machines with the hardware present, e.g. _gpu_. |
| classes | True/**False** | Whether more than one type of this co-processor is available |
| include\_more\_capable | **True**/False | Whether to automatically request all classes that are more capable than the requested class. This requires the _class\_constraints_ option to be set to True and for your SLURM cluster to be set up with GPU features/constraints |
| class\_types | Configuration dictionary | This contains the definition of the GPU classes... |
| | _Key_ | |
| | class selector | This is the letter (or word) that is used to select this class of co-processor from the fsl\_sub commandline. For CUDA devices you may consider using the card name e.g. A100.|
| | resource | This is the name of the SLURM GRES 'type' or contraint that will be used to select this GPU family.
| | doc | The description that appears in the fsl\_sub help text about this device.
| | capability | An integer defining the feature set of the device, your most basic device should be given the value 1 and more capable devices higher values, e.g. GTX = 1, Kelper = 2, Pascal = 3, Volta = 4.
| default\_class | _Class type key_ | The _class selector_ for the class to assign jobs to where a class has not been specified in the fsl\_sub call. For FSL tools that automatically submit to CUDA queues you should aim to select one that has good double-precision performance (K40|80, P100, V100, A100) and ensure all higher capability devices also have good double-precision.
| class\_constraints | **True**/False | Whether your SLURM cluster is configured with co-processor selection features. If so this should be set to true and the co-processor _resource_ and class _resource_ strings set appropriately to match the available values. |
| presence\_test | _Program path_ (**nvidia-smi** for CUDA) | The name of a program that can be used to look for this coprocessor type, for example nvidia-smi for CUDA devices. Program needs to return non-zero exit status if there are no available coprocessors.

### Queue Definitions

Slurm refers to queues as partitions. The example configuration should contain definitions for the automatically discovered partitions but you should review these, in particular any warnings generated.
To query SLURM for queue information you can use the following SLURM commands.

To get a list of all available partitions use:

> sinfo -s -o %P

Then the details for a queue can be obtained with:

> sinfo -p _partitionname_ -O 'CPUs,MaxCPUsPerNode,Memory,Time,NodeHost'

This will return details for every node within that partition. The queue definition should then be setup as follows:

| Key | Value type | Description
|-----|------------|-------------|
| time | integer in minutes | The _TIMELIMIT_ column reports in days-hours:minutes:seconds, this needs converting to minutes. Provide the maximum value observed, but if there are multiple values you should consider enabling job time notification so that SLURM can select the correct node.||
| max\_size | integer in GB | This is the maximum permitted memory on a node. This is usually reported by SLURM in MB, so for example 63000 should be configured as 63 (GB). It is equal to the maximum _MEMORY_ value reported. Once again, if there are multiple node types you should turn on RAM nofitication so that nodes can be correctly selected.
| max\_slots | _CPU_ contains the number of CPUs (threads) available on each node. Set this option to the maximum number reported. |
| slot\_size | **Null**/integer in GB | This is largely meaningless on SLURM and left at None. If you find that you need to get fsl\_sub to split your job into multiple threads to achieve your memory requirements then set this to the figure provided by your cluster manager. |
| group | integer | (Optional) All partitions with the same group number will be considered together when scheduling, typically this would be all queues with the same run time but differing memory/core counts.|

Where a partition has obvious GRES or features that define GPUs a proposed GPU configuration will be added as comments to the start of the queue definition. You should review this, create/update the coproc_opts>cuda record with the information in the comments and then this section can be uncommented to enable GPU support.

#### Compound Queues

Some clusters may be configured with multiple variants of the same partition, e.g. short.a, short.b, with each queue having different hardware, perhaps CPU generation or maximum memory or memory available per slot. To maximise scheduling options you can define compound queues which have the configuration of the least capable constituent. To define a compound queue, the queue name (key of the YAML dictionary) should be a comma separated list of queue names (no space).
