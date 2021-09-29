# fsl_sub plugin for:
#  * Slurm
import datetime
import logging
import os
import subprocess as sp
from collections import defaultdict
from ruamel.yaml.comments import CommentedMap
from shutil import which

from fsl_sub.exceptions import (
    BadSubmission,
    MissingConfiguration,
    GridOutputError,
    UnknownJobId,
)
from fsl_sub.config import (
    method_config,
    coprocessor_config,
)
import fsl_sub.consts
from fsl_sub.coprocessors import (
    coproc_get_module
)
from fsl_sub.shell_modules import loaded_modules
from fsl_sub.utils import (
    affirmative,
    split_ram_by_slots,
    human_to_ram,
    parse_array_specifier,
    bash_cmd,
    fix_permissions,
    flatten_list,
    job_script,
    write_wrapper,
    update_envvar_list,
)
from .version import PLUGIN_VERSION


METHOD_NAME = 'slurm'


def plugin_version():
    return PLUGIN_VERSION


def qtest():
    '''Command that confirms method is available'''
    return _sinfo_cmd()


def _sinfo_cmd():
    '''Command that queries queue configuration'''
    qconf = which('sinfo')
    if qconf is None:
        raise BadSubmission("Cannot find Slurm software")
    return qconf


def _qsub_cmd():
    '''Command that submits a job'''
    qsub = which('sbatch')
    if qsub is None:
        raise BadSubmission("Cannot find Slurm software")
    return qsub


def _sacctmgr_cmd():
    '''Command that manages accounts'''
    sacctmgr = which('sacctmgr')
    if sacctmgr is None:
        raise BadSubmission("Cannot find Slurm software")
    return sacctmgr


def _sacct_cmd():
    '''Command that queries job stats'''
    sacct = which('sacct')
    if sacct is None:
        raise BadSubmission("Cannot find Slurm software")
    return sacct


def _squeue_cmd():
    '''Command that queries running job stats'''
    squeue = which('squeue')
    if squeue is None:
        raise BadSubmission("Cannot find Slurm software")
    return squeue


def queue_exists(qname, qtest=None):
    '''Does qname exist'''
    if qtest is None:
        qtest = which('sinfo')
        if qtest is None:
            raise BadSubmission("Cannot find Slurm software")
    if '@' in qname:
        qlist = []
        for q in qname.split(','):
            qlist.append(q.split('@')[0])
        qname = ','.join(qlist)
    try:
        output = sp.run(
            [qtest, '--noheader', '-p', qname],
            stdout=sp.PIPE,
            check=True, universal_newlines=True)
    except sp.CalledProcessError:
        raise BadSubmission("Cannot run Slurm software")
    if output.stdout:
        return True
    else:
        return False


def already_queued():
    '''Is this a running SLURM job?'''
    return ('SLURM_JOB_ID' in os.environ.keys() or 'SLURM_JOBID' in os.environ.keys())


def qdel(job_id):
    '''Deletes a job - returns a tuple, output, return code'''
    scancel = which('scancel')
    if scancel is None:
        raise BadSubmission("Cannot find Slurm software")
    result = sp.run(
        [scancel, str(job_id), ],
        universal_newlines=True,
        stdout=sp.PIPE, stderr=sp.STDOUT
    )
    return (result.stdout, result.returncode)


def _slurm_option(opt):
    return "#SBATCH " + opt


def _get_logger():
    return logging.getLogger('fsl_sub.' + __name__)


def submit(
        command,
        job_name,
        queue,
        threads=1,
        array_task=False,
        jobhold=None,
        array_hold=None,
        array_limit=None,
        array_specifier=None,
        parallel_env=None,
        jobram=None,
        jobtime=None,
        resources=None,
        ramsplit=False,
        priority=None,
        mail_on=None,
        mailto=None,
        logdir=None,
        coprocessor=None,
        coprocessor_toolkit=None,
        coprocessor_class=None,
        coprocessor_class_strict=False,
        coprocessor_multi=1,
        usescript=False,
        architecture=None,
        requeueable=True,
        project=None,
        export_vars=None,
        keep_jobscript=False):
    '''Submits the job to a SLURM cluster
    Requires:

    command - list containing command to run
                or the file name of the array task file.
                If array_specifier is given then this must be
                a list containing the command to run.
    job_name - Symbolic name for task
    queue - Queue to submit to

    Optional:
    array_task - is the command is an array task (defaults to False)
    jobhold - id(s) of jobs to hold for (string or list)
    array_hold - complex hold string, integer or list
    array_limit - limit concurrently scheduled array
            tasks to specified number
    array_specifier - n[-m[:s]] n subtasks or starts at n, ends at m with
            a step of s
    parallelenv - parallel environment name
    jobram - RAM required by job (total of all threads)
    jobtime - time (in minutes for task)
    requeueable - may a job be requeued if a node fails
    resources - list of resource request strings
    ramsplit - break tasks into multiple slots to meet RAM constraints
    priority - job priority - not supported
    mail_on - mail user on 'a'bort or reschedule, 'b'egin, 'e'nd,
            's'uspended, 'n'o mail
    mailto - email address to receive job info
    logdir - directory to put log files in
    coprocessor - name of coprocessor required
    coprocessor_toolkit - coprocessor toolkit version
    coprocessor_class - class of coprocessor required
    coprocessor_class_strict - whether to choose only this class
            or all more capable
    coprocessor_multi - how many coprocessors you need (or
            complex description) (string)
    usescript - queue config is defined in script
    project - which account to associate this job with
    export_vars - list of environment variables to preserve for job
            ignored if job is copying complete environment
    keep_jobscript - whether to generate (if not configured already) and keep
            a wrapper script for the job
    '''

    logger = _get_logger()

    if command is None:
        raise BadSubmission(
            "Must provide command line or array task file name")
    if not isinstance(command, list):
        raise BadSubmission(
            "Internal error: command argument must be a list"
        )

    # Can't just have export_vars=[] in function definition as the list is mutable so subsequent calls
    # will return the updated list!
    if export_vars is None:
        export_vars = []
    my_export_vars = list(export_vars)

    mconf = defaultdict(lambda: False, method_config(METHOD_NAME))
    qsub = _qsub_cmd()
    command_args = []
    extra_lines = []

    modules = []
    if logdir is None:
        logdir = os.getcwd()
    if isinstance(resources, str):
        resources = [resources, ]

    array_map = {
        'FSLSUB_JOB_ID_VAR': 'SLURM_JOB_ID',
        'FSLSUB_ARRAYTASKID_VAR': 'SLURM_ARRAY_TASK_ID',
        'FSLSUB_ARRAYSTARTID_VAR': 'SLURM_ARRAY_TASK_MIN',
        'FSLSUB_ARRAYENDID_VAR': 'SLURM_ARRAY_TASK_MAX',
        'FSLSUB_ARRAYSTEPSIZE_VAR': 'SLURM_ARRAY_TASK_STEP',
        'FSLSUB_ARRAYCOUNT_VAR': 'SLURM_ARRAY_TASK_COUNT',
        'FSLSUB_NSLOTS': 'SLURM_NPROCS'
    }

    if queue is None:
        raise BadSubmission("Queue not specified")
    if type(queue) == str:
        if ',' in queue:
            queues = queue.split(',')
        else:
            queues = [queue, ]
    elif type(queue) == list:
        queues = queue
    pure_queues = [q.split('@')[0] for q in queues]

    gres = []
    if usescript:
        if len(command) > 1:
            raise BadSubmission(
                "Command should be a grid submission script (no arguments)")
        use_jobscript = False
        keep_jobscript = False
    else:
        use_jobscript = mconf.get('use_jobscript', True)
        if not keep_jobscript:
            keep_jobscript = mconf.get('keep_jobscript', False)
        if keep_jobscript:
            use_jobscript = True
        # Check Parallel Environment is available
        if parallel_env:
            command_args.extend(
                ['--ntasks-per-node', str(threads), ])

        for var, value in array_map.items():
            if not value:
                value = '""'
            update_envvar_list(my_export_vars, '='.join((var, value)))
        if mconf.get('copy_environment', False):
            my_export_vars.insert(0, 'ALL')

        my_evars = []
        if my_export_vars:
            for var in my_export_vars:
                if '=' in var:
                    vname, vvalue = var.split('=', 1)
                    # Check if there is a comma or space in the env-var value, if so add it to my_complex_vars
                    if any(x in vvalue for x in [',', ' ']):

                        if (
                                (vvalue.startswith('"') and vvalue.endswith('"'))
                                or (vvalue.startswith("'") and vvalue.endswith("'"))):
                            my_evars.append(var)
                        else:
                            my_evars.append("{0}='{1}'".format(vname, vvalue))
                    else:
                        my_evars.append(var)
                else:
                    my_evars.append(var)

            command_args.append(
                '='.join(('--export', ','.join(my_evars)))
            )

        def cp_class_item(cp, cpclass, item):
            return cp['class_types'][cpclass][item]

        if coprocessor is not None:

            # Setup the coprocessor
            cpconf = coprocessor_config(coprocessor)

            gres_items = [cpconf['resource'], str(coprocessor_multi), ]

            cpclasses = []

            if cpconf.get('classes', False) and coprocessor_class is None:
                coprocessor_class = cpconf.get('default_class', None)
                cpclasses.append(cp_class_item(cpconf, coprocessor_class, 'resource'))

            if cpconf.get('classes', False):
                class_constraint = cpconf.get('class_constraint', False)

                if class_constraint:
                    if cpconf.get('include_more_capable', True) and not coprocessor_class_strict:
                        cp_capability = cp_class_item(cpconf, coprocessor_class, 'capability')
                        base_list = [
                            a for a in cpconf['class_types'].keys()
                            if cpconf['class_types'][a]['capability'] > cp_capability]
                        [cpclasses.append(
                            cpconf['class_types'][a]['resource']) for a in
                            sorted(
                                base_list,
                                key=lambda x:
                                cpconf['class_types'][x]['capability'])
                            if a not in cpclasses]
                    else:
                        cpclasses.append(cp_class_item(cpconf, coprocessor_class, 'resource'))

                    constraints = [":".join((class_constraint, cpc)) for cpc in cpclasses]
                    command_args.append('='.join(
                        ('--constraint', '"{0}"'.format('|'.join(constraints)))
                    ))
                else:
                    if len(cpclasses) == 1:
                        gres_items.insert(1, cpclasses[0])
            else:
                if len(cpclasses) == 1:
                    gres_items.insert(1, cpclasses[0])

            gres.append(":".join(gres_items))

        if resources:
            gres.append(','.join(resources))

        if gres:
            command_args.append('='.join(
                ('--gres', ",".join(gres))
            ))

        if logdir == '/dev/null':
            command_args.append(['-o', logdir, ])
            command_args.append(['-e', logdir, ])
        else:
            logs = {}
            for lf in ['o', 'e', ]:
                if array_task:
                    logtemplate = '{0}.{1}%A.%a'
                else:
                    logtemplate = '{0}.{1}%j'
                logs[lf] = os.path.join(
                    logdir,
                    logtemplate.format(
                        job_name.replace(' ', '_'),
                        lf)
                )
            command_args.append(['-o', logs['o'], ])
            command_args.append(['-e', logs['e'], ])

        hold_state = 'afterok'
        if array_task and array_hold is not None:
            if mconf['array_holds']:
                # Requires Slurm 16.05
                jobhold = array_hold
                hold_state = 'aftercorr'
            else:
                jobhold = array_hold

        if jobhold:
            if isinstance(jobhold, (list, tuple, )):
                parents = ':'.join([str(a).replace('.', '_') for a in jobhold])
            elif isinstance(jobhold, str):
                parents = jobhold
            elif isinstance(jobhold, int):
                parents = str(jobhold)
            else:
                raise BadSubmission(
                    "jobhold is of unsupported type " + type(jobhold))
            command_args.append(
                "=".join(
                    (
                        '--dependency',
                        hold_state + ':' + parents)
                )
            )

        if array_task is not None:
            # ntasks%array_limit
            if mconf['array_limits'] and array_limit:
                array_limit_modifier = "%{}".format(array_limit)
            else:
                array_limit_modifier = ""

        if jobram:
            # Slurm defaults to dividing up the task into multiple cpu
            # requests, automatically reducing memory per cpu value.
            # However, we have already done this, so we need to
            # reduce the RAM requirements.

            if ramsplit:
                jobram = split_ram_by_slots(jobram, threads)
                # mem-per-cpu if dividing RAM up, otherwise mem
            if mconf['notify_ram_usage']:
                command_args.append(
                    '='.join((
                        '--mem-per-cpu',
                        str(jobram) + fsl_sub.consts.RAMUNITS
                    ))
                )
        try:
            no_set_tlimit = (os.environ['FSLSUB_NOTIMELIMIT'] == '1' or affirmative(os.environ['FSLSUB_NOTIMELIMIT']))
        except Exception:
            no_set_tlimit = False
        if jobtime:
            if mconf['set_time_limit'] and not no_set_tlimit:
                command_args.append(['-t', jobtime])
        if mconf['mail_support']:
            if mailto:
                command_args.extend(['-M', mailto, ])
                if not mail_on:
                    mail_on = mconf['mail_mode']
                if mail_on not in mconf['mail_modes']:
                    raise BadSubmission("Unrecognised mail mode")
                command_args.append(
                    '='.join((
                        '--mail-type',
                        ','.join(mconf['mail_mode'][mail_on]),
                    ))
                )
        command_args.append(
            '='.join((
                '--job-name', job_name, ))
        )
        # Set current working directory
        command_args.append(
            '='.join((
                '--chdir', os.getcwd()
            ))
        )
        hlist = []
        for q in queues:
            if '@' in q:
                qname, qhost = q.split('@')
                hlist.append(qhost)

        command_args.append(['-p', ','.join(pure_queues)])
        if hlist:
            command_args.append(['-w', ','.join(hlist), ])
        command_args.append('--parsable')

        if requeueable:
            command_args.append('--requeue')

        if project is not None:
            command_args.append('--account ' + project)

        if array_task:
            # Submit array task file
            if array_specifier:
                (
                    array_start,
                    array_end,
                    array_stride
                ) = parse_array_specifier(array_specifier)
                if not array_start:
                    raise BadSubmission("array_specifier doesn't make sense")
                array_spec = "{0}". format(array_start)
                if array_end:
                    array_spec += "-{0}".format(array_end)
                if array_stride:
                    array_spec += ":{0}".format(array_stride)
                command_args.append(
                    "=".join('--array', "{0}{1}".format(
                        array_spec,
                        array_limit_modifier)))
            else:
                with open(command[0], 'r') as cmd_f:
                    array_slots = len(cmd_f.readlines())
                command_args.append(
                    "=".join((
                        '--array', "1-{0}{1}".format(
                            array_slots,
                            array_limit_modifier))))

    logger.info("slurm_args: " + " ".join(
        [str(a) for a in command_args if a != qsub]))

    bash = bash_cmd()

    if array_task and not array_specifier:
        logger.info("executing array task")
    else:
        if usescript:
            logger.info("executing cluster script")
        else:
            if array_specifier:
                logger.info("excuting array task {0}-{1}:{2}".format(
                    array_start,
                    array_end,
                    array_stride
                ))
            else:
                logger.info("executing single task")

    logger.info(" ".join([str(a) for a in command_args]))
    logger.debug(type(command_args))
    logger.debug(command_args)

    if array_task and not array_specifier:
        extra_lines.extend([
            '',
            'the_command=$(sed -n -e "${{SLURM_ARRAY_TASK_ID}}p" {0})'.format(command[0]),
            '',
        ])
        command = ['exec', bash, '-c', '"$the_command"', ]
        command_args = command_args if use_jobscript else []
        use_jobscript = True

    modules_paths = None
    if mconf.get('preserve_modules', True):
        modules = loaded_modules()
        if coprocessor_toolkit:
            cp_module = coproc_get_module(coprocessor, coprocessor_toolkit)
            if cp_module is not None:
                modules.append(cp_module)
        modules_paths = mconf.get('add_module_paths', [])
    js_lines = job_script(
        command, command_args,
        '#SBATCH', (METHOD_NAME, plugin_version()),
        modules=modules, extra_lines=extra_lines, modules_paths=modules_paths)
    logger.debug('\n'.join(js_lines))
    if keep_jobscript:
        wrapper_name = write_wrapper(js_lines)
        logger.debug(wrapper_name)
        command_args = [wrapper_name]
        logger.debug("Calling fix_permissions " + str(0o755))
        fix_permissions(wrapper_name, 0o755)
    else:
        if not usescript:
            command_args = []
        else:
            command_args = flatten_list(command_args)
            command_args.extend(command)

    command_args.insert(0, qsub)

    if keep_jobscript:
        result = sp.run(
            command_args, universal_newlines=True,
            stdout=sp.PIPE, stderr=sp.PIPE)
    else:
        result = sp.run(
            command_args,
            input='\n'.join(js_lines),
            universal_newlines=True,
            stdout=sp.PIPE, stderr=sp.PIPE
        )
    if result.returncode != 0:
        raise BadSubmission(result.stderr)
    job_words = result.stdout.split(';')
    try:
        job_id = int(job_words[0].split('.')[0])
    except ValueError:
        raise GridOutputError("Grid output was " + result.stdout)

    if keep_jobscript:
        new_name = os.path.join(
            os.getcwd(),
            '_'.join(('wrapper', str(job_id))) + '.sh'
        )
        try:
            logger.debug("Renaming wrapper to " + new_name)
            os.rename(
                wrapper_name,
                new_name
            )
        except OSError:
            logger.warn("Unable to preserve wrapper script")
    return job_id


def _default_config_file():
    return os.path.join(
        os.path.realpath(os.path.dirname(__file__)),
        'fsl_sub_slurm.yml')


def default_conf():
    '''Returns a string containing the default configuration for this
    cluster plugin.'''

    try:
        with open(_default_config_file()) as d_conf_f:
            d_conf = d_conf_f.read()
    except FileNotFoundError as e:
        raise MissingConfiguration("Unable to find default configuration file: " + str(e))
    return d_conf


def job_status(job_id, sub_job_id=None):
    '''Return details for the job with given ID.

    details holds a dict with following info:
        id
        name
        sub_time
        tasks (dict keyed on sub-task ID):
            status:
                fsl_sub.consts.QUEUED
                fsl_sub.consts.RUNNING
                fsl_sub.consts.FINISHED
                fsl_sub.consts.FAILED
                fsl_sub.consts.HELD
            start_time
            end_time
        '''

    # Look for running jobs
    if isinstance(job_id, str):
        if '.' in job_id:
            if sub_job_id is None:
                job_id, sub_job_id = job_id.split('.')
                sub_job_id = int(sub_job_id)
            else:
                job_id, _ = job_id.split('.')
        job_id = int(job_id)
    if isinstance(sub_job_id, str):
        sub_job_id = int(sub_job_id)

    try:
        job_details = _job(job_id, sub_job_id=sub_job_id)

    except UnknownJobId:
        raise
    except Exception as e:
        raise GridOutputError from e

    return job_details


def _get_sacct(job_id, sub_job_id=None):
    sacct_args = [
        '--parsable2',
        '--noheader',
        '--units=M',
        '--duplicate',
        '--format', ','.join((
            'JobID',
            'JobName',
            'Submit',
            'Start',
            'End',
            'State',
            'ExitCode',
        )
        )
    ]
    if sub_job_id is not None:
        job = ".".join(str(job_id), str(sub_job_id))
    else:
        job = str(job_id)
    sacct = [_sacct_cmd()]
    sacct.extend(['-j', job])
    sacct.extend(sacct_args)
    output = None
    try:
        sacct_barsv = sp.run(
            sacct,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            check=True, universal_newlines=True)
        output = sacct_barsv.stdout
    except FileNotFoundError:
        raise BadSubmission(
            "Slurm software may not be correctly installed")
    except sp.CalledProcessError as e:
        raise GridOutputError(e.stderr)
    if not output:
        raise UnknownJobId

    job = {}
    job['tasks'] = {}

    for line in output.splitlines():
        fields = line.split('|')
        if '.' in fields[0]:
            continue
        if '_' in fields[0]:
            # An array task
            jid, sjid = fields[0].split('_')
            jid, sjid = (int(jid), int(sjid))
        else:
            jid, sjid = fields[0], 1
            jid, sjid = (int(jid), int(sjid))

        job['id'] = jid

        if sjid not in job['tasks']:
            job['tasks'][sjid] = {}

        task = job['tasks'][sjid]
        exit_status = int(fields[6].split(':')[0])
        if exit_status != 0:
            task['status'] = fsl_sub.consts.FAILED
        else:
            status = fields[5]
            if status == 'REQUEUED':
                task['status'] = fsl_sub.consts.REQUEUED
            elif status == 'SUSPENDED':
                task['status'] = fsl_sub.consts.SUSPENDED
            elif status in ['RUNNING', 'RESIZING']:
                task['status'] = fsl_sub.consts.RUNNING
            elif status == 'PENDING':
                task['status'] = fsl_sub.consts.QUEUED
            elif status == 'COMPLETED':
                task['status'] = fsl_sub.consts.FINISHED
            else:
                task['status'] = fsl_sub.consts.FAILED
        task['start_time'] = _sacct_datetimestamp(fields[3])
        task['end_time'] = _sacct_datetimestamp(fields[4])

        job['sub_time'] = _sacct_datetimestamp(fields[2])
        job['name'] = fields[1]

    return job


def _sacct_datetimestamp(output):
    if output == 'Unknown':
        return None
    return datetime.datetime.strptime(output, '%Y-%m-%dT%H:%M:%S')


def _sacct_timestamp_seconds(output):
    if output == 'Unknown':
        return None

    duration = 0
    if '-' in output:
        duration += int(output.split('-')[0]) * 86400
        output = output.split('-')[1]
    index = output.count(':')
    for sub_time in output.split(':'):
        if '.' in sub_time:
            stime = float(sub_time)
        else:
            stime = int(sub_time)
        duration += stime * 60**index
        index -= 1
    return float(duration)


def _get_data(getter, job_id, sub_job_id=None):
    try:
        job_info = getter(job_id, sub_job_id)
    except UnknownJobId:
        return None

    if sub_job_id is not None:
        for s_task in job_info[job_id]['tasks'].keys():
            if s_task != sub_job_id:
                del job_info[job_id]['tasks'][s_task]

    return job_info


def _job(job_id, sub_job_id=None):
    job_info = _get_data(_get_sacct, job_id, sub_job_id)
    return job_info


def project_list():
    '''This returns a list of recognised projects (or accounts) that a job
    can be allocated to (e.g. for billing or fair share allocation)'''
    accounts_cmd = _sacctmgr_cmd()
    try:
        accounts_out = sp.run(
            [accounts_cmd, '-P', '-r', '-n', 'list', 'Account', ],
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            check=True, universal_newlines=True
        )
    except FileNotFoundError:
        raise BadSubmission(
            "SLURM software may not be correctly installed")
    except sp.CalledProcessError as e:
        raise GridOutputError(e.stderr)
    return [a[0] for a in accounts_out.stdout.split('|')]


def _get_queues(sinfo=None):
    '''Return list of partition names'''
    if sinfo is None:
        sinfo = _sinfo_cmd()
    try:
        result = sp.run(
            [sinfo, '-s', '-h', '-o', '%P', ],
            stdout=sp.PIPE,
            stderr=sp.DEVNULL,
            check=True, universal_newlines=True)
    except (FileNotFoundError, sp.CalledProcessError, ):
        raise BadSubmission(
            "SLURM software may not be correctly installed")
    queues = []
    default = None
    for q in result.stdout.splitlines():
        if '*' in q:
            q = q.strip('*')
            default = q
        queues.append(q)

    return (queues, default)


def _get_queue_features(queue, sinfo=None):
    if sinfo is None:
        sinfo = _sinfo_cmd()

    features = defaultdict(list)
    try:
        result = sp.run(
            [sinfo, '-h', '-p', queue, '-o', '%f', ],
            stdout=sp.PIPE,
            stderr=sp.DEVNULL,
            check=True, universal_newlines=True)
    except FileNotFoundError:
        raise BadSubmission(
            "SLURM software may not be correctly installed")
    except sp.CalledProcessError:
        raise BadSubmission(
            "Queue {0} not found!".format(queue))
    for fl in result.stdout.splitlines():
        fs = fl.split(',')
        for f in fs:
            f = f.strip()
            if f != '':
                if ':' in f:
                    feature, value = f.split(':')
                    features[feature].append(value)
                else:
                    features[f] = []
    return features


def _get_queue_gres(queue, sinfo=None):
    logger = _get_logger()

    if sinfo is None:
        sinfo = _sinfo_cmd()

    gres = defaultdict(list)
    try:
        result = sp.run(
            [sinfo, '-h', '-p', queue, '-o', '%G', ],
            stdout=sp.PIPE,
            stderr=sp.DEVNULL,
            check=True, universal_newlines=True)
    except FileNotFoundError:
        raise BadSubmission(
            "SLURM software may not be correctly installed")
    except sp.CalledProcessError:
        raise BadSubmission(
            "Queue {0} not found!".format(queue))
    for gres_line in result.stdout.splitlines():
        for sub_g in gres_line.split(','):
            # The output of this is hard to parse and I can't find it
            # documented anywhere. This is a best effort!
            sub_g = sub_g.replace('(null)', '')
            if '(' in sub_g:
                # Strip off extra info from last field which should be number
                sub_g, _ = sub_g.split('(')
            fields = sub_g.split(':')
            if len(fields) == 2:
                # name:number
                gres[fields[0]].append(('-', _get_gres_count(fields[1])))
            elif len(fields) == 3:
                # name:type:number
                gres[fields[0]].append((fields[1], _get_gres_count(fields[2])))
            elif len(fields) == 4:
                # name:type:'no_consume':number
                gres[fields[0]].append((fields[1], _get_gres_count(fields[3])))
            else:
                # Not recognized - ignore
                logger.warn("Unrecognized GRES field: %s (full: %s)", sub_g, gres_line)
    return gres


def _get_gres_count(gres_count):
    try:
        count = int(gres_count)
    except ValueError:
        count = human_to_ram(gres_count, output='B', as_int=True)
    return count


def _get_queue_info(queue, sinfo=None):
    '''Return dictionary of queue info'''
    if sinfo is None:
        sinfo = _sinfo_cmd()
    mconfig = method_config(METHOD_NAME)
    fields = [
        'CPUs',
        'MaxCPUsPerNode',
        'Memory',
        'Time',
        'NodeHost',
    ]
    sinfo_cmd = [sinfo, '-p', queue, '-h', '-O', ]
    sinfo_cmd.append(",".join(fields))
    try:
        result = sp.run(
            sinfo_cmd,
            stdout=sp.PIPE,
            stderr=sp.DEVNULL,
            check=True, universal_newlines=True)
    except FileNotFoundError:
        raise BadSubmission(
            "SLURM software may not be correctly installed")
    except sp.CalledProcessError:
        raise BadSubmission(
            "Queue {0} not found!".format(queue))

    qvariants = []
    output = result.stdout
    conf_lines = output.splitlines()
    for cl in conf_lines:
        (cpus, maxcpus, memory, qtime, _) = cl.split()
        cpus = int(cpus)
        try:
            maxcpus = int(maxcpus)
            cpus = max(cpus, maxcpus)
        except ValueError:
            pass
        memory = int(memory)
        if not mconfig['memory_in_gb']:
            memory = memory // 1000  # Memory reported in MB
        if qtime == "infinite":
            qtime = "365-23:59:59"
        qtime = _day_time_minutes(qtime)
        qvariants.append((cpus, memory, qtime, ))

    qdef = {'qname': queue, 'cpus': None, 'memory': None, 'qtime': None, }
    comments = []
    for qv in qvariants:
        cpus, memory, qtime = qv
        if qdef['cpus'] is not None:
            if qdef['cpus'] != cpus:
                _add_comment(
                    comments,
                    "Partition contains nodes with different numbers of CPUs")
            if qdef['cpus'] < cpus:
                qdef['cpus'] = cpus
        else:
            qdef['cpus'] = cpus
        if qdef['memory'] is not None:
            if qdef['memory'] != memory:
                _add_comment(
                    comments,
                    "Partition contains nodes with different amounts of memory,"
                    " consider switching on RAM nofitication")
            if qdef['memory'] < memory:
                qdef['memory'] = memory
        else:
            qdef['memory'] = memory
        if qdef['qtime'] is not None:
            if qdef['qtime'] != qtime:
                _add_comment(
                    comments,
                    "Partition contains nodes with differing maximum run times,"
                    " consider switching on time notification")
            if qdef['qtime'] < qtime:
                qdef['qtime'] = qtime
        else:
            qdef['qtime'] = qtime

    return qdef, comments


def _day_time_minutes(dayt):
    '''Convert D-HH:MM:SS to minutes'''
    if '-' in dayt:
        (days, sub_day) = dayt.split('-')
        days = int(days)
    else:
        sub_day = dayt
        days = 0

    if ':' not in sub_day:
        minutes = 0
        hours = 0
        seconds = sub_day
    elif sub_day.count(':') == 1:
        (minutes, seconds) = sub_day.split(':')
        hours = 0
    else:
        (hours, minutes, seconds) = sub_day.split(':')
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)

    if days == 0 and hours == 0 and minutes == 0 and seconds != 0:
        minutes += 1
    return days * (24 * 60) + hours * 60 + minutes


def _add_comment(comments, comment):
    if comment not in comments:
        comments.append(comment)


def build_queue_defs():
    '''Return YAML suitable for configuring queues'''
    logger = _get_logger()

    try:
        queue_list, default = _get_queues()
    except BadSubmission as e:
        logger.error('Unable to query SLURM: ' + str(e))
        return ('', [])
    q_base = CommentedMap()
    q_base['queues'] = CommentedMap()
    queues = q_base['queues']
    for q in sorted(queue_list):
        qinfo, comments = _get_queue_info(q)
        gres = _get_queue_gres(q)
        features = _get_queue_features(q)
        queues[qinfo['qname']] = CommentedMap()
        qd = queues[qinfo['qname']]
        queues.yaml_add_eol_comment("Queue name", qinfo['qname'], column=0)
        add_key_comment = qd.yaml_add_eol_comment
        for coproc_m in ('gpu', 'cuda', 'phi', ):
            if coproc_m in q:
                _add_comment(
                    comments,
                    "'Queue name looks like it might be a queue supporting co-processors."
                    " Cannot auto-configure.'"
                )
        qd['time'] = qinfo['qtime']
        add_key_comment('Maximum job run time in minutes', 'time', column=0)
        qd['max_slots'] = qinfo['cpus']
        add_key_comment("Maximum number of threads/slots on a queue", 'max_slots', column=0)
        qd['max_size'] = qinfo['memory']
        add_key_comment("Maximum RAM size of a job in " + fsl_sub.consts.RAMUNITS + 'B', 'max_size', column=0)
        qd['slot_size'] = None
        add_key_comment(
            "Slot size is normally irrelevant on SLURM "
            "- set this to memory (in {0}B) per thread if required".format(fsl_sub.consts.RAMUNITS),
            'slot_size')
        if 'gpu' in gres.keys():

            _add_comment(
                comments,
                "Partion has a GRES 'gpu' that might indicate the presence of GPUs,"
                " see below for possible configuration")
            _add_comment(
                comments,
                "coproc: cuda 'resource' would be 'gpu' and associated class "
                "resources ('-' if none):quantities would be:")
            for res_p in gres['gpu']:
                _add_comment(
                    comments, ":".join((str(res_p[0]), str(res_p[1]))))
            gpu_matches = [(k, v) for k, v in features.items() if 'gpu' in k]
            if gpu_matches:
                _add_comment(
                    comments,
                    "Partition has features that look like GPU resources, these might be usable as constraints"
                )
                for constraint, options in gpu_matches:
                    if options:
                        _add_comment(
                            comments,
                            "If using constraints the coproc: cuda 'resource' could be {0} "
                            "and associated class 'resource's would be {1}".format(
                                constraint, ','.join(options))
                        )
            qty = max([q[1] for q in gres['gpu']])
            _add_comment(comments, 'copros:')
            _add_comment(comments, '  cuda: # CUDA Co-processor available')
            _add_comment(comments, '    max_quantity: ' + str(qty) + ' # Maximum available per node')
            _add_comment(comments, '    classes: # List of classes (if classes supported)')
            for res_p in sorted(list(set(gres['gpu']))):
                _add_comment(comments, '      - ' + res_p[0])
            _add_comment(comments, '    exclusive: False # Does this only run jobs requiring this co-processor?')

        _add_comment(comments, "default: true # Is this the default partition?")
        _add_comment(comments, 'priority: 1 # Priority in group - higher wins')
        _add_comment(comments, 'group: 1 # Group partitions with the same integer then order by priority')

        for w in comments:
            queues.yaml_set_comment_before_after_key(qinfo['qname'], after=w)

    return q_base
