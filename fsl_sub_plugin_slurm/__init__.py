# fsl_sub plugin for:
#  * Slurm
import datetime
import logging
import os
import subprocess as sp
from collections import defaultdict
from shutil import which

from fsl_sub.exceptions import (
    BadSubmission,
    BadConfiguration,
    MissingConfiguration,
    GridOutputError,
    UnknownJobId,
)
from fsl_sub.config import (
    method_config,
    coprocessor_config,
    read_config,
)
import fsl_sub.consts
from fsl_sub.coprocessors import (
    coproc_get_module
)
from fsl_sub.shell_modules import loaded_modules
from fsl_sub.utils import (
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
    return qconf_cmd()


def qconf_cmd():
    '''Command that queries queue configuration'''
    qconf = which('scontrol')
    if qconf is None:
        raise BadSubmission("Cannot find Slurm software")
    return qconf


def qstat_cmd():
    '''Command that queries queue state'''
    qstat = which('squeue')
    if qstat is None:
        raise BadSubmission("Cannot find Slurm software")
    return qstat


def qsub_cmd():
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


def queue_exists(qname, qtest=None):
    '''Does qname exist'''
    if qtest is None:
        qtest = which('sinfo')
        if qtest is None:
            raise BadSubmission("Cannot find Slurm software")
    found = []
    for q in qname.split(','):
        q = q.split('@')[0]
        try:
            output = sp.run(
                [qtest, '--noheader', '-p', qname],
                stdout=sp.PIPE,
                check=True, universal_newlines=True)
        except sp.CalledProcessError:
            raise BadSubmission("Cannot run Slurm software")
        if output.stdout:
            found.append(True)
    if found:
        return all(found)
    else:
        return False


def already_queued():
    '''Is this a running SLURM job?'''
    return ('SLURM_JOB_ID' in os.environ.keys() or 'SLURM_JOBID' in os.environ.keys())


def sacct_cmd():
    '''Command that queries job stats'''
    sacct = which('sacct')
    if sacct is None:
        raise BadSubmission("Cannot find Slurm software")
    return sacct


def squeue_cmd():
    '''Command that queries running job stats'''
    squeue = which('squeue')
    if squeue is None:
        raise BadSubmission("Cannot find Slurm software")
    return squeue


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
        export_vars=[],
        keep_jobscript=None):
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
    array_hold - complex hold string
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
    priority - job priority (0-1023)
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
    my_export_vars = list(export_vars)
    if command is None:
        raise BadSubmission(
            "Must provide command line or array task file name")
    if not isinstance(command, list):
        raise BadSubmission(
            "Internal error: command argument must be a list"
        )
    mconf = defaultdict(lambda: False, method_config(METHOD_NAME))
    qsub = qsub_cmd()
    command_args = []

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
    }

    gres = []
    if usescript:
        if len(command) > 1:
            raise BadSubmission(
                "Command should be a grid submission script (no arguments)")
        use_jobscript = False
        keep_jobscript = False
    else:
        use_jobscript = mconf.get('use_jobscript', True)
        if keep_jobscript is None:
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
        if my_export_vars:
            command_args.append(
                '='.join(('--export', ','.join(my_export_vars)))
            )

        if coprocessor is not None:
            # Setup the coprocessor
            cpconf = coprocessor_config(coprocessor)

            if cpconf['classes']:
                available_classes = cpconf['class_types']
                if coprocessor_class is None:
                    coprocessor_class = cpconf['default_class']

                try:
                    copro_class = available_classes[
                        coprocessor_class][
                            'resource']
                except KeyError:
                    raise BadSubmission("Unrecognised coprocessor class")
                if (not coprocessor_class_strict and
                        cpconf['include_more_capable'] and
                        cpconf['slurm_constraints']):
                    copro_capability = available_classes[
                        coprocessor_class]['capability']
                    base_list = [
                        a for a in cpconf['class_types'].keys() if
                        cpconf['class_types'][a]['capability'] >=
                        copro_capability]
                    copro_class = '|'.join(
                        [
                            cpconf['class_types'][a]['resource'] for a in
                            sorted(
                                base_list,
                                key=lambda x:
                                cpconf['class_types'][x]['capability'])
                        ]
                    )
                    command_args.append(
                        '='.join(
                            ('--constraint', '"{}"'.format(copro_class)))
                    )
                    gres.append(
                        ":".join(
                            (cpconf['resource'], str(coprocessor_multi))))
            else:
                gres.append(
                    ":".join(
                        (
                            cpconf['resource'],
                            str(coprocessor_multi))
                    )
                )

        # Job priorities can only be set by admins

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
            for l in ['o', 'e', ]:
                if array_task:
                    logtemplate = '{0}.{1}%A.%a'
                else:
                    logtemplate = '{0}.{1}%j'
                logs[l] = os.path.join(
                    logdir,
                    logtemplate.format(
                        job_name.replace(' ', '_'),
                        l)
                )
            command_args.append(['-o', logs['o'], ])
            command_args.append(['-e', logs['e'], ])

        hold_state = 'afterok'
        if array_task and array_hold:
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
                        '--dependancy',
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
            # request, automatically reducing memory per cpu value
            # However, we have already done this, so we need to
            # reduce the RAM requirements.

            if ramsplit:
                jobram = split_ram_by_slots(jobram, threads)
                # mem-per-cpu if dividing RAM up, otherwise mem
            ram_units = read_config()['ram_units']

            # RAM is specified in megabytes
            try:
                mem_in_mb = human_to_ram(
                    jobram,
                    units=ram_units,
                    output="M")
            except ValueError:
                raise BadConfiguration("ram_units not one of P, T, G, M, K")
            if mconf['notify_ram_usage']:
                command_args.append(
                    '='.join((
                        '--mem-per-cpu',
                        str(int(mem_in_mb))
                    ))
                )

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
        qlist = []
        hlist = []
        for q in queue.split(','):
            if '@' in q:
                qname, qhost = q.split('@')
                qlist.append(qname)
                hlist.append(qhost)
            else:
                qlist.append(q)

        command_args.append(['-p', ','.join(qlist)])
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

    extra_lines = []
    if array_task and not array_specifier:
        extra_lines = [
            '',
            'the_command=$(sed -n -e "${{SLURM_ARRAY_TASK_ID}}p" {0})'.format(command),
            '',
        ]
        command = ['exec', bash, '-c', '"$the_command"', ]
        command_args = command_args if use_jobscript else []
        use_jobscript = True

    if mconf.get('preserve_modules', True):
        modules = loaded_modules()
        if coprocessor_toolkit:
            cp_module = coproc_get_module(coprocessor, coprocessor_toolkit)
            if cp_module is not None:
                modules.append(cp_module)
    js_lines = job_script(
        command, command_args,
        '#SBATCH', (METHOD_NAME, plugin_version()),
        modules=modules, extra_lines=extra_lines)
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


def example_conf():
    '''Returns a string containing the example configuration for this
    cluster plugin.'''

    try:
        with open(_default_config_file()) as e_conf_f:
            e_conf = e_conf_f.read()
    except FileNotFoundError as e:
        raise MissingConfiguration("Unable to find example configuration file: " + str(e))
    return e_conf


def job_status(job_id, sub_job_id=None):
    '''Return details for the job with given ID.

    details holds a dict with following info:
        id
        name
        script (if available)
        arguments (if available)
        # sub_state: fsl_sub.consts.NORMAL|RESTARTED|SUSPENDED
        submission_time
        tasks (dict keyed on sub-task ID):
            status:
                fsl_sub.consts.QUEUED
                fsl_sub.consts.RUNNING
                fsl_sub.consts.FINISHED
                fsl_sub.consts.FAILEDNQUEUED
                fsl_sub.consts.HELD
            start_time
            end_time
            sub_time
            exit_status
            error_message
            maxmemory (in Mbytes)
        parents (if available)
        children (if available)
        job_directory (if available)

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
        job_details = _running_job(job_id, sub_job_id)
        if job_details:
            return job_details
        else:
            job_details = _finished_job(job_id, sub_job_id)

    except UnknownJobId as e:
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
            'UserCPU',
            'SystemCPU',
            'State',
            'ExitCode',
            'MaxVMSize')
        )
    ]
    if sub_job_id is not None:
        job = ".".join(str(job_id), str(sub_job_id))
    else:
        job = str(job_id)
    sacct = [sacct_cmd()]
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
    for line in output.splitlines():
        fields = line.split('|')
        stage = ''
        if '_' in fields[0]:
            # An array task
            jid, sjid = fields[0].split('_')
            jid = int(jid)
            if '.batch' in sjid:
                sjid, stage = sjid.split('.')
            jid, sjid = (int(jid), int(sjid))
        else:
            jid, sjid = fields[0], 1
            if '.batch' in jid:
                jid, stage = jid.split('.')
            jid, sjid = (int(jid), int(sjid))

        job['id'] = jid
        job['script'] = None
        job['arguments'] = None
        job['submission_time'] = _sacct_datetimestamp(fields[2])
        job['parents'] = None
        job['children'] = None
        job['job_directory'] = None
        job['tasks'] = {}

        if sjid not in job['tasks']:
            job['tasks'][sjid] = {}

        task = job['tasks'][sjid]
        task['exit_status'] = int(fields[8].split(':')[0])
        if task['exit_status'] != 0:
            task['status'] = fsl_sub.consts.FAILED
        else:
            status = fields[7]
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
        task['error_message'] = None
        task['end_time'] = _sacct_datetimestamp(fields[4])
        task['start_time'] = _sacct_datetimestamp(fields[3])
        task['sub_time'] = _sacct_datetimestamp(fields[2])
        task['utime'] = _sacct_timestamp_seconds(fields[5])
        task['stime'] = _sacct_timestamp_seconds(fields[6])
        # Slurm STATES should be mapped to our status...

        if stage == '':
            job['name'] = fields[1]
        else:
            task['maxmemory'] = human_to_ram(fields[-1], as_int=False)

    return job


def _get_squeue(job_id, sub_job_id=None):
    fn = {
        'jobid': '%F',
        'array_id': '%K',
        'name': '%j',
        'sub_time': '%V',
        'start_time': '%S',
        'end_time': '%e',
        'command': '%o',
        'depends': '%E',
        'state': '%t',
        'job_directory': '%Z'
    }
    # This is untestable as the dictionary order will change each time run.
    fn_order = list(fn.keys())
    fn_index = {a: fn_order.index(a) for a in fn.keys()}
    squeue_args = [
        '--noheader',
        '-o',
        ','.join(fn.order)
    ]

    if sub_job_id is not None:
        job = "_".join(str(job_id), str(sub_job_id))
    else:
        job = str(job_id)
    squeue = [squeue_cmd()]
    squeue.extend(['--jobs=', job])
    squeue.extend(squeue_args)

    output = None
    try:
        squeue_csv = sp.run(
            squeue,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            check=True, universal_newlines=True)
        output = squeue_csv.stdout
    except FileNotFoundError:
        raise BadSubmission(
            "Slurm software may not be correctly installed")
    except sp.CalledProcessError as e:
        if "Invalid job id specified" in e.stderr:
            raise UnknownJobId
        else:
            raise GridOutputError(e.stderr)

    if not output:
        raise UnknownJobId

    jobs = {}
    for line in output:
        fields = line.split('|')
        jid = fields[fn_index('jobid')]
        sjid = fields[fn_index['array_id']]
        if jid not in jobs:
            jobs[jid] = {}
            jobs[jid]['tasks'] = {}

        if sjid not in jobs[jid]['tasks']:
            jobs[jid]['tasks'][sjid] = {}

        jobs[jid]['script'] = fields[fn_index['command']]
        task = jobs[jid]['tasks'][sjid]
        task['exit_status'] = None
        status = fields[fn_index['state']]
        if status == 'RQ':
            task['status'] = fsl_sub.consts.REQUEUED
        elif status in ['RD', 'RH']:
            task['status'] = fsl_sub.consts.HELD
        elif status == 'S':
            task['status'] = fsl_sub.consts.SUSPENDED
        elif status in ['R', 'CG', 'RS']:
            task['status'] = fsl_sub.consts.RUNNING
        elif status == 'CF':
            task['status'] = fsl_sub.consts.STARTING
        elif status == 'PD':
            task['status'] = fsl_sub.consts.QUEUED
        elif status == 'CD':
            task['status'] = fsl_sub.consts.FINISHED
        else:
            task['status'] = fsl_sub.consts.FAILED
        task['error_message'] = None
        task['end_time'] = _sacct_datetimestamp(fields[fn_index['end_time']])
        task['start_time'] = _sacct_datetimestamp(
            fields[fn_index['start_time']])
        task['sub_time'] = _sacct_datetimestamp(fields[fn_index['sub_time']])
        task['utime'] = None
        task['stime'] = None
        task['maxmemory'] = None
    return jobs


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


def _running_job(job_id, sub_job_id=None):
    job_info = _get_data(_get_squeue, job_id, sub_job_id)
    return job_info


def _finished_job(job_id, sub_job_id=None):
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
            "Grid Engine software may not be correctly installed")
    except sp.CalledProcessError as e:
        raise GridOutputError(e.stderr)
    return [a[0] for a in accounts_out.stdout.split('|')]


def build_queue_defs():
    '''Not currently implemented'''
    return ''
