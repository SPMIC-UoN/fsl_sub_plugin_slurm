#!/usr/bin/env python
import copy
import datetime
import io
import os
import subprocess
import tempfile
import unittest
import fsl_sub_plugin_slurm

from collections import defaultdict
from ruamel.yaml import YAML
from unittest.mock import (patch, )

import fsl_sub.consts
from fsl_sub.exceptions import (
    BadSubmission,
    UnknownJobId
)
from fsl_sub.utils import (
    yaml_repr_none,
)

conf_dict = YAML(typ='safe').load('''---
method_opts:
    slurm:
        memory_in_gb: False
        queues: True
        copy_environment: True
        mail_support: True
        mail_modes:
            b:
                - BEGIN
            e:
                - END
            a:
                - FAIL
                - REQUEUE
            f:
                - ALL
            n:
                - NONE
        mail_mode: a
        set_time_limit: False
        array_holds: True
        array_limit: True
        preserve_modules: True
        add_module_paths: []
        keep_jobscript: False
copro_opts:
    cuda:
        resource: gpu
        classes: True
        class_resource: gputype
        class_types:
            K:
                resource: k80
                doc: Kepler. ECC, double- or single-precision workloads
                capability: 2
            P:
                resource: p100
                doc: >
                    Pascal. ECC, double-, single- and half-precision
                    workloads
                capability: 3
        default_class: K
        include_more_capable: True
        uses_modules: True
        module_parent: cuda
        no_binding: True
        class_constriant: True
''')
mconf_dict = conf_dict['method_opts']['slurm']


class TestSlurmUtils(unittest.TestCase):
    def test__sacct_datetimestamp(self):
        self.assertEqual(
            fsl_sub_plugin_slurm._sacct_datetimestamp('2018-06-04T10:30:30'),
            datetime.datetime(2018, 6, 4, 10, 30, 30)
        )

    def test__sacct_timstamp_seconds(self):
        self.assertEqual(
            fsl_sub_plugin_slurm._sacct_timestamp_seconds('10:10:10.10'),
            36610.1
        )
        self.assertEqual(
            fsl_sub_plugin_slurm._sacct_timestamp_seconds('5-10:10:10.10'),
            468610.1
        )
        self.assertEqual(
            fsl_sub_plugin_slurm._sacct_timestamp_seconds('1:10.10'),
            70.1
        )

    def test__day_time_minutes(self):
        self.assertEqual(
            fsl_sub_plugin_slurm._day_time_minutes('1-00:00:00'),
            24 * 60
        )
        self.assertEqual(
            fsl_sub_plugin_slurm._day_time_minutes('1-00:01:00'),
            24 * 60 + 1
        )
        self.assertEqual(
            fsl_sub_plugin_slurm._day_time_minutes('0-00:01:00'),
            1
        )
        self.assertEqual(
            fsl_sub_plugin_slurm._day_time_minutes('0-00:00:01'),
            1
        )
        self.assertEqual(
            fsl_sub_plugin_slurm._day_time_minutes('0-01:00:00'),
            60
        )
        self.assertEqual(
            fsl_sub_plugin_slurm._day_time_minutes('0-01:01:00'),
            60 + 1
        )
        self.assertEqual(
            fsl_sub_plugin_slurm._day_time_minutes('10:00'),
            10
        )
        self.assertEqual(
            fsl_sub_plugin_slurm._day_time_minutes('10'),
            1
        )

    def test__add_comment(self):
        comments = []
        comments.append('A comment')
        newcomments = list(comments)
        fsl_sub_plugin_slurm._add_comment(
            newcomments, 'A comment'
        )
        self.assertListEqual(
            comments,
            newcomments
        )
        fsl_sub_plugin_slurm._add_comment(
            newcomments, 'Another comment'
        )
        self.assertListEqual(
            ['A comment', 'Another comment', ],
            newcomments
        )


class TestslurmFinders(unittest.TestCase):
    @patch('fsl_sub_plugin_slurm.which', autospec=True)
    def test_qstat(self, mock_which):
        bin_path = '/usr/bin/squeue'
        with self.subTest("Test 1"):
            mock_which.return_value = bin_path
            self.assertEqual(
                bin_path,
                fsl_sub_plugin_slurm._squeue_cmd()
            )
        mock_which.reset_mock()
        with self.subTest("Test 2"):
            mock_which.return_value = None
            self.assertRaises(
                fsl_sub_plugin_slurm.BadSubmission,
                fsl_sub_plugin_slurm._squeue_cmd
            )

    @patch('fsl_sub_plugin_slurm.which', autospec=True)
    def test_qsub(self, mock_which):
        bin_path = '/usr/bin/sbatch'
        with self.subTest("Test 1"):
            mock_which.return_value = bin_path
            self.assertEqual(
                bin_path,
                fsl_sub_plugin_slurm._qsub_cmd()
            )
        mock_which.reset_mock()
        with self.subTest("Test 2"):
            mock_which.return_value = None
            self.assertRaises(
                fsl_sub_plugin_slurm.BadSubmission,
                fsl_sub_plugin_slurm._qsub_cmd
            )

    @patch('fsl_sub_plugin_slurm.sp.run', autospec=True)
    def test_queue_exists(self, mock_spr):
        bin_path = '/usr/bin/sinfo'
        qname = 'myq'
        with patch(
            'fsl_sub_plugin_slurm.which',
            autospec=True,
            return_value=None
        ):
            with self.subTest("No sinfo"):
                self.assertRaises(
                    BadSubmission,
                    fsl_sub_plugin_slurm.queue_exists,
                    '123'
                )
        mock_spr.reset_mock()
        with patch(
                'fsl_sub_plugin_slurm.which',
                return_value=bin_path):

            with self.subTest("Test 1"):
                mock_spr.return_value = subprocess.CompletedProcess(
                    [bin_path, '--noheader', '-p', qname],
                    stdout='',
                    returncode=0
                )
                self.assertFalse(
                    fsl_sub_plugin_slurm.queue_exists(qname)
                )
                mock_spr.assert_called_once_with(
                    [bin_path, '--noheader', '-p', qname],
                    stdout=subprocess.PIPE,
                    check=True,
                    universal_newlines=True)
            mock_spr.reset_mock()
            with self.subTest("Test 2"):
                mock_spr.return_value = subprocess.CompletedProcess(
                    [bin_path, '--noheader', '-p', qname],
                    stdout='A   up  5-00:00:00  1   idle    anode1234\n',
                    returncode=0
                )
                self.assertTrue(
                    fsl_sub_plugin_slurm.queue_exists(qname, bin_path)
                )


@patch('fsl_sub.utils.VERSION', '1.0.0')
@patch(
    'fsl_sub_plugin_slurm.os.getcwd',
    autospec=True, return_value='/Users/testuser')
@patch(
    'fsl_sub_plugin_slurm._qsub_cmd',
    autospec=True, return_value='/usr/bin/sbatch'
)
@patch('fsl_sub_plugin_slurm.split_ram_by_slots', autospec=True)
@patch('fsl_sub_plugin_slurm.coprocessor_config', autospec=True)
@patch('fsl_sub_plugin_slurm.sp.run', autospec=True)
class TestSubmit(unittest.TestCase):
    def setUp(self):
        self.ww = tempfile.NamedTemporaryFile(
            mode='w+t',
            delete=False)
        self.now = datetime.datetime.now()
        self.strftime = datetime.datetime.strftime
        self.bash = '/bin/bash'
        self.config = copy.deepcopy(conf_dict)
        self.mconfig = self.config['method_opts']['slurm']
        self.patch_objects = {
            'fsl_sub.utils.datetime': {'autospec': True, },
            'fsl_sub_plugin_slurm.plugin_version': {'autospec': True, 'return_value': '2.0.0', },
            'fsl_sub_plugin_slurm.loaded_modules': {'autospec': True, 'return_value': ['mymodule', ], },
            'fsl_sub_plugin_slurm.bash_cmd': {'autospec': True, 'return_value': self.bash, },
            'fsl_sub_plugin_slurm.write_wrapper': {'autospec': True, 'side_effect': self.w_wrapper},
            'fsl_sub_plugin_slurm.method_config': {'autospec': True, 'return_value': self.mconfig, },
        }
        self.patch_dict_objects = {}
        self.patches = {}
        for p, kwargs in self.patch_objects.items():
            self.patches[p] = patch(p, **kwargs)
        self.mocks = {}
        for o, p in self.patches.items():
            self.mocks[o] = p.start()

        self.dict_patches = {}
        for p, kwargs in self.patch_dict_objects.items():
            self.dict_patches[p] = patch.dict(p, **kwargs)

        for o, p in self.dict_patches.items():
            self.mocks[o] = p.start()
        self.mocks['fsl_sub.utils.datetime'].datetime.now.return_value = self.now
        self.mocks['fsl_sub.utils.datetime'].datetime.strftime = self.strftime
        self.addCleanup(patch.stopall)

    def TearDown(self):
        self.ww.close()
        os.unlink(self.ww.name)
        patch.stopall()

    plugin = fsl_sub_plugin_slurm

    def w_wrapper(self, content):
        for lf in content:
            self.ww.write(lf + '\n')
        return self.ww.name

    def test_empty_submit(
            self, mock_sprun, mock_cpconf,
            mock_srbs, mock_qsub,
            mock_getcwd):
        self.assertRaises(
            self.plugin.BadSubmission,
            self.plugin.submit,
            None, None, None
        )

    def test_submit_basic(
            self, mock_sprun, mock_cpconf,
            mock_srbs, mock_qsub,
            mock_getcwd):
        job_name = 'test_job'
        queue = 'a.q'
        cmd = ['./acmd', 'arg1', 'arg2', ]
        logdir = os.getcwd()
        jid = 12345
        qsub_out = str(jid)
        with self.subTest("Slurm"):
            expected_cmd = ['/usr/bin/sbatch']
            expected_script = (
                '#!' + self.bash + '''

#SBATCH --export=ALL,'''
                '''FSLSUB_JOB_ID_VAR=SLURM_JOB_ID,'''
                '''FSLSUB_ARRAYTASKID_VAR=SLURM_ARRAY_TASK_ID,'''
                '''FSLSUB_ARRAYSTARTID_VAR=SLURM_ARRAY_TASK_MIN,'''
                '''FSLSUB_ARRAYENDID_VAR=SLURM_ARRAY_TASK_MAX,'''
                '''FSLSUB_ARRAYSTEPSIZE_VAR=SLURM_ARRAY_TASK_STEP,'''
                '''FSLSUB_ARRAYCOUNT_VAR=SLURM_ARRAY_TASK_COUNT,'''
                '''FSLSUB_NSLOTS=SLURM_NPROCS
#SBATCH -o {0}.o%j
#SBATCH -e {0}.e%j
#SBATCH --job-name={1}
#SBATCH --chdir={5}
#SBATCH -p {2}
#SBATCH --parsable
#SBATCH --requeue
module load mymodule
# Built by fsl_sub v.1.0.0 and fsl_sub_plugin_slurm v.2.0.0
# Command line: fsl_sub -q {2} {3}
# Submission time (H:M:S DD/MM/YYYY): {4}

{3}
'''.format(
                    os.path.join(logdir, job_name),
                    job_name, queue, ' '.join(cmd),
                    self.now.strftime("%H:%M:%S %d/%m/%Y"),
                    logdir)
            )
            mock_sprun.return_value = subprocess.CompletedProcess(
                expected_cmd, 0,
                stdout=qsub_out, stderr=None)
            with patch('fsl_sub.utils.sys.argv', ['fsl_sub', '-q', 'a.q', './acmd', 'arg1', 'arg2']):
                self.assertEqual(
                    jid,
                    self.plugin.submit(
                        command=cmd,
                        job_name=job_name,
                        queue=queue
                    )
                )
            mock_sprun.assert_called_once_with(
                expected_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                input=expected_script
            )

    def test_submit_singlehost(
            self, mock_sprun, mock_cpconf,
            mock_srbs, mock_qsub,
            mock_getcwd):
        job_name = 'test_job'
        queue = 'a.q@host1'
        cmd = ['./acmd', 'arg1', 'arg2', ]
        logdir = os.getcwd()
        jid = 12345
        qsub_out = str(jid)
        with self.subTest("Slurm"):
            expected_cmd = ['/usr/bin/sbatch']
            expected_script = (
                '#!' + self.bash + '''

#SBATCH --export=ALL,'''
                '''FSLSUB_JOB_ID_VAR=SLURM_JOB_ID,'''
                '''FSLSUB_ARRAYTASKID_VAR=SLURM_ARRAY_TASK_ID,'''
                '''FSLSUB_ARRAYSTARTID_VAR=SLURM_ARRAY_TASK_MIN,'''
                '''FSLSUB_ARRAYENDID_VAR=SLURM_ARRAY_TASK_MAX,'''
                '''FSLSUB_ARRAYSTEPSIZE_VAR=SLURM_ARRAY_TASK_STEP,'''
                '''FSLSUB_ARRAYCOUNT_VAR=SLURM_ARRAY_TASK_COUNT,'''
                '''FSLSUB_NSLOTS=SLURM_NPROCS
#SBATCH -o {0}.o%j
#SBATCH -e {0}.e%j
#SBATCH --job-name={1}
#SBATCH --chdir={7}
#SBATCH -p {2}
#SBATCH -w {5}
#SBATCH --parsable
#SBATCH --requeue
module load mymodule
# Built by fsl_sub v.1.0.0 and fsl_sub_plugin_slurm v.2.0.0
# Command line: fsl_sub -q {6} {3}
# Submission time (H:M:S DD/MM/YYYY): {4}

{3}
'''.format(
                    os.path.join(logdir, job_name),
                    job_name, 'a.q', ' '.join(cmd),
                    self.now.strftime("%H:%M:%S %d/%m/%Y"),
                    'host1',
                    queue,
                    logdir)
            )
            mock_sprun.return_value = subprocess.CompletedProcess(
                expected_cmd, 0,
                stdout=qsub_out, stderr=None)
            with patch('fsl_sub.utils.sys.argv', ['fsl_sub', '-q', queue, './acmd', 'arg1', 'arg2']):
                self.assertEqual(
                    jid,
                    self.plugin.submit(
                        command=cmd,
                        job_name=job_name,
                        queue=queue
                    )
                )
            mock_sprun.assert_called_once_with(
                expected_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                input=expected_script
            )

    def test_submit_multiq(
            self, mock_sprun, mock_cpconf,
            mock_srbs, mock_qsub,
            mock_getcwd):
        job_name = 'test_job'
        queue = ['a.q', 'b.q', ]
        cmd = ['./acmd', 'arg1', 'arg2', ]
        logdir = os.getcwd()
        jid = 12345
        qsub_out = str(jid)
        with self.subTest("Slurm"):
            expected_cmd = ['/usr/bin/sbatch']
            expected_script = (
                '#!' + self.bash + '''

#SBATCH --export=ALL,'''
                '''FSLSUB_JOB_ID_VAR=SLURM_JOB_ID,'''
                '''FSLSUB_ARRAYTASKID_VAR=SLURM_ARRAY_TASK_ID,'''
                '''FSLSUB_ARRAYSTARTID_VAR=SLURM_ARRAY_TASK_MIN,'''
                '''FSLSUB_ARRAYENDID_VAR=SLURM_ARRAY_TASK_MAX,'''
                '''FSLSUB_ARRAYSTEPSIZE_VAR=SLURM_ARRAY_TASK_STEP,'''
                '''FSLSUB_ARRAYCOUNT_VAR=SLURM_ARRAY_TASK_COUNT,'''
                '''FSLSUB_NSLOTS=SLURM_NPROCS
#SBATCH -o {0}.o%j
#SBATCH -e {0}.e%j
#SBATCH --job-name={1}
#SBATCH --chdir={5}
#SBATCH -p {2}
#SBATCH --parsable
#SBATCH --requeue
module load mymodule
# Built by fsl_sub v.1.0.0 and fsl_sub_plugin_slurm v.2.0.0
# Command line: fsl_sub -q {2} {3}
# Submission time (H:M:S DD/MM/YYYY): {4}

{3}
'''.format(
                    os.path.join(logdir, job_name),
                    job_name, ','.join(queue), ' '.join(cmd),
                    self.now.strftime("%H:%M:%S %d/%m/%Y"),
                    logdir
                )
            )
            mock_sprun.return_value = subprocess.CompletedProcess(
                expected_cmd, 0,
                stdout=qsub_out, stderr=None)
            with patch('fsl_sub.utils.sys.argv', ['fsl_sub', '-q', ','.join(queue), './acmd', 'arg1', 'arg2']):
                self.assertEqual(
                    jid,
                    self.plugin.submit(
                        command=cmd,
                        job_name=job_name,
                        queue=','.join(queue)
                    )
                )
            mock_sprun.assert_called_once_with(
                expected_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                input=expected_script
            )

    def test_submit_multiq_multih(
            self, mock_sprun, mock_cpconf,
            mock_srbs, mock_qsub,
            mock_getcwd):
        job_name = 'test_job'
        queue = ['a.q@host1', 'b.q', ]
        cmd = ['./acmd', 'arg1', 'arg2', ]
        logdir = os.getcwd()
        jid = 12345
        qsub_out = str(jid)
        with self.subTest("Slurm"):
            expected_cmd = ['/usr/bin/sbatch']
            expected_script = (
                '#!' + self.bash + '''

#SBATCH --export=ALL,'''
                '''FSLSUB_JOB_ID_VAR=SLURM_JOB_ID,'''
                '''FSLSUB_ARRAYTASKID_VAR=SLURM_ARRAY_TASK_ID,'''
                '''FSLSUB_ARRAYSTARTID_VAR=SLURM_ARRAY_TASK_MIN,'''
                '''FSLSUB_ARRAYENDID_VAR=SLURM_ARRAY_TASK_MAX,'''
                '''FSLSUB_ARRAYSTEPSIZE_VAR=SLURM_ARRAY_TASK_STEP,'''
                '''FSLSUB_ARRAYCOUNT_VAR=SLURM_ARRAY_TASK_COUNT,'''
                '''FSLSUB_NSLOTS=SLURM_NPROCS
#SBATCH -o {0}.o%j
#SBATCH -e {0}.e%j
#SBATCH --job-name={1}
#SBATCH --chdir={7}
#SBATCH -p {5}
#SBATCH -w {4}
#SBATCH --parsable
#SBATCH --requeue
module load mymodule
# Built by fsl_sub v.1.0.0 and fsl_sub_plugin_slurm v.2.0.0
# Command line: fsl_sub -q {6} {2}
# Submission time (H:M:S DD/MM/YYYY): {3}

{2}
'''.format(
                    os.path.join(logdir, job_name),
                    job_name,
                    ' '.join(cmd),
                    self.now.strftime("%H:%M:%S %d/%m/%Y"),
                    'host1',
                    'a.q,b.q',
                    ','.join(queue),
                    logdir
                )
            )
            mock_sprun.return_value = subprocess.CompletedProcess(
                expected_cmd, 0,
                stdout=qsub_out, stderr=None)
            with patch('fsl_sub.utils.sys.argv', ['fsl_sub', '-q', ','.join(queue), './acmd', 'arg1', 'arg2']):
                self.assertEqual(
                    jid,
                    self.plugin.submit(
                        command=cmd,
                        job_name=job_name,
                        queue=','.join(queue)
                    )
                )
            mock_sprun.assert_called_once_with(
                expected_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                input=expected_script
            )

    def test_submit_multiq_multih2(
            self, mock_sprun, mock_cpconf,
            mock_srbs, mock_qsub,
            mock_getcwd):
        job_name = 'test_job'
        queue = ['a.q@host1', 'b.q@host2', ]
        cmd = ['./acmd', 'arg1', 'arg2', ]
        logdir = os.getcwd()
        jid = 12345
        qsub_out = str(jid)
        with self.subTest("Slurm"):
            expected_cmd = ['/usr/bin/sbatch']
            expected_script = (
                '#!' + self.bash + '''

#SBATCH --export=ALL,'''
                '''FSLSUB_JOB_ID_VAR=SLURM_JOB_ID,'''
                '''FSLSUB_ARRAYTASKID_VAR=SLURM_ARRAY_TASK_ID,'''
                '''FSLSUB_ARRAYSTARTID_VAR=SLURM_ARRAY_TASK_MIN,'''
                '''FSLSUB_ARRAYENDID_VAR=SLURM_ARRAY_TASK_MAX,'''
                '''FSLSUB_ARRAYSTEPSIZE_VAR=SLURM_ARRAY_TASK_STEP,'''
                '''FSLSUB_ARRAYCOUNT_VAR=SLURM_ARRAY_TASK_COUNT,'''
                '''FSLSUB_NSLOTS=SLURM_NPROCS
#SBATCH -o {0}.o%j
#SBATCH -e {0}.e%j
#SBATCH --job-name={1}
#SBATCH --chdir={7}
#SBATCH -p {2}
#SBATCH -w {5}
#SBATCH --parsable
#SBATCH --requeue
module load mymodule
# Built by fsl_sub v.1.0.0 and fsl_sub_plugin_slurm v.2.0.0
# Command line: fsl_sub -q {6} {3}
# Submission time (H:M:S DD/MM/YYYY): {4}

{3}
'''.format(
                    os.path.join(logdir, job_name),
                    job_name, ','.join([q.split('@')[0] for q in queue]),
                    ' '.join(cmd),
                    self.now.strftime("%H:%M:%S %d/%m/%Y"),
                    'host1,host2',
                    ','.join(queue),
                    logdir
                )
            )
            mock_sprun.return_value = subprocess.CompletedProcess(
                expected_cmd, 0,
                stdout=qsub_out, stderr=None)
            with patch('fsl_sub.utils.sys.argv', ['fsl_sub', '-q', ','.join(queue), './acmd', 'arg1', 'arg2']):
                self.assertEqual(
                    jid,
                    self.plugin.submit(
                        command=cmd,
                        job_name=job_name,
                        queue=','.join(queue)
                    )
                )
            mock_sprun.assert_called_once_with(
                expected_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                input=expected_script
            )

    def test_project_submit(
            self, mock_sprun, mock_cpconf,
            mock_srbs, mock_qsub,
            mock_getcwd):
        job_name = 'test_job'
        queue = 'a.q'
        project = 'Aproject'
        cmd = ['./acmd', 'arg1', 'arg2', ]
        logdir = os.getcwd()
        jid = 12345
        qsub_out = str(jid)
        with self.subTest("No projects"):
            w_conf = copy.deepcopy(self.config)
            w_conf['method_opts']['slurm']['projects'] = True
            self.mocks['fsl_sub_plugin_slurm.method_config'].return_value = w_conf['method_opts']['slurm']
            expected_cmd = ['/usr/bin/sbatch']
            expected_script = (
                '#!' + self.bash + '''

#SBATCH --export=ALL,'''
                '''FSLSUB_JOB_ID_VAR=SLURM_JOB_ID,'''
                '''FSLSUB_ARRAYTASKID_VAR=SLURM_ARRAY_TASK_ID,'''
                '''FSLSUB_ARRAYSTARTID_VAR=SLURM_ARRAY_TASK_MIN,'''
                '''FSLSUB_ARRAYENDID_VAR=SLURM_ARRAY_TASK_MAX,'''
                '''FSLSUB_ARRAYSTEPSIZE_VAR=SLURM_ARRAY_TASK_STEP,'''
                '''FSLSUB_ARRAYCOUNT_VAR=SLURM_ARRAY_TASK_COUNT,'''
                '''FSLSUB_NSLOTS=SLURM_NPROCS
#SBATCH -o {0}.o%j
#SBATCH -e {0}.e%j
#SBATCH --job-name={1}
#SBATCH --chdir={5}
#SBATCH -p {2}
#SBATCH --parsable
#SBATCH --requeue
module load mymodule
# Built by fsl_sub v.1.0.0 and fsl_sub_plugin_slurm v.2.0.0
# Command line: fsl_sub -q {2} {3}
# Submission time (H:M:S DD/MM/YYYY): {4}

{3}
'''.format(
                    os.path.join(logdir, job_name),
                    job_name, queue, ' '.join(cmd),
                    self.now.strftime("%H:%M:%S %d/%m/%Y"),
                    logdir
                )
            )
            mock_sprun.return_value = subprocess.CompletedProcess(
                expected_cmd, 0,
                stdout=qsub_out, stderr=None)
            with patch('fsl_sub.utils.sys.argv', ['fsl_sub', '-q', 'a.q', './acmd', 'arg1', 'arg2']):
                self.assertEqual(
                    jid,
                    self.plugin.submit(
                        command=cmd,
                        job_name=job_name,
                        queue=queue
                    )
                )
            mock_sprun.assert_called_once_with(
                expected_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                input=expected_script
            )
        mock_sprun.reset_mock()
        self.mocks['fsl_sub_plugin_slurm.method_config'].return_value = mconf_dict
        with self.subTest("With Project"):
            expected_cmd = ['/usr/bin/sbatch']
            expected_script = (
                '#!' + self.bash + '''

#SBATCH --export=ALL,'''
                '''FSLSUB_JOB_ID_VAR=SLURM_JOB_ID,'''
                '''FSLSUB_ARRAYTASKID_VAR=SLURM_ARRAY_TASK_ID,'''
                '''FSLSUB_ARRAYSTARTID_VAR=SLURM_ARRAY_TASK_MIN,'''
                '''FSLSUB_ARRAYENDID_VAR=SLURM_ARRAY_TASK_MAX,'''
                '''FSLSUB_ARRAYSTEPSIZE_VAR=SLURM_ARRAY_TASK_STEP,'''
                '''FSLSUB_ARRAYCOUNT_VAR=SLURM_ARRAY_TASK_COUNT,'''
                '''FSLSUB_NSLOTS=SLURM_NPROCS
#SBATCH -o {0}.o%j
#SBATCH -e {0}.e%j
#SBATCH --job-name={1}
#SBATCH --chdir={6}
#SBATCH -p {2}
#SBATCH --parsable
#SBATCH --requeue
#SBATCH --account {3}
module load mymodule
# Built by fsl_sub v.1.0.0 and fsl_sub_plugin_slurm v.2.0.0
# Command line: fsl_sub -q {2} {4}
# Submission time (H:M:S DD/MM/YYYY): {5}

{4}
'''.format(
                    os.path.join(logdir, job_name),
                    job_name, queue, project, ' '.join(cmd),
                    self.now.strftime("%H:%M:%S %d/%m/%Y"),
                    logdir)
            )
            mock_sprun.return_value = subprocess.CompletedProcess(
                expected_cmd, 0,
                stdout=qsub_out, stderr=None)
            with patch('fsl_sub.utils.sys.argv', ['fsl_sub', '-q', 'a.q', './acmd', 'arg1', 'arg2']):
                self.assertEqual(
                    jid,
                    self.plugin.submit(
                        command=cmd,
                        job_name=job_name,
                        queue=queue,
                        project='Aproject'
                    )
                )
            mock_sprun.assert_called_once_with(
                expected_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                input=expected_script
            )
        mock_sprun.reset_mock()
        with self.subTest("With modules path"):
            w_conf = copy.deepcopy(self.config)
            w_conf['method_opts']['slurm']['projects'] = True
            w_conf['method_opts']['slurm']['add_module_paths'] = ['/usr/local/shellmodules']
            self.mocks['fsl_sub_plugin_slurm.method_config'].return_value = w_conf['method_opts']['slurm']
            expected_cmd = ['/usr/bin/sbatch']
            expected_script = (
                '#!' + self.bash + '''

#SBATCH --export=ALL,'''
                '''FSLSUB_JOB_ID_VAR=SLURM_JOB_ID,'''
                '''FSLSUB_ARRAYTASKID_VAR=SLURM_ARRAY_TASK_ID,'''
                '''FSLSUB_ARRAYSTARTID_VAR=SLURM_ARRAY_TASK_MIN,'''
                '''FSLSUB_ARRAYENDID_VAR=SLURM_ARRAY_TASK_MAX,'''
                '''FSLSUB_ARRAYSTEPSIZE_VAR=SLURM_ARRAY_TASK_STEP,'''
                '''FSLSUB_ARRAYCOUNT_VAR=SLURM_ARRAY_TASK_COUNT,'''
                '''FSLSUB_NSLOTS=SLURM_NPROCS
#SBATCH -o {0}.o%j
#SBATCH -e {0}.e%j
#SBATCH --job-name={1}
#SBATCH --chdir={6}
#SBATCH -p {2}
#SBATCH --parsable
#SBATCH --requeue
#SBATCH --account {3}
MODULEPATH=/usr/local/shellmodules:$MODULEPATH
module load mymodule
# Built by fsl_sub v.1.0.0 and fsl_sub_plugin_slurm v.2.0.0
# Command line: fsl_sub -q {2} {4}
# Submission time (H:M:S DD/MM/YYYY): {5}

{4}
'''.format(
                    os.path.join(logdir, job_name),
                    job_name, queue, project, ' '.join(cmd),
                    self.now.strftime("%H:%M:%S %d/%m/%Y"),
                    logdir)
            )
            mock_sprun.return_value = subprocess.CompletedProcess(
                expected_cmd, 0,
                stdout=qsub_out, stderr=None)
            with patch('fsl_sub.utils.sys.argv', ['fsl_sub', '-q', 'a.q', './acmd', 'arg1', 'arg2']):
                self.assertEqual(
                    jid,
                    self.plugin.submit(
                        command=cmd,
                        job_name=job_name,
                        queue=queue,
                        project='Aproject'
                    )
                )
            mock_sprun.assert_called_once_with(
                expected_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                input=expected_script
            )

        mock_sprun.reset_mock()
        with self.subTest("GPU without constraints"):
            w_conf = copy.deepcopy(self.config)
            w_conf['method_opts']['slurm']['projects'] = True
            w_conf['method_opts']['slurm']['add_module_paths'] = ['/usr/local/shellmodules']
            w_conf['copro_opts']['cuda']['class_constraint'] = False
            self.mocks['fsl_sub_plugin_slurm.method_config'].return_value = w_conf['method_opts']['slurm']
            mock_cpconf.return_value = w_conf['copro_opts']['cuda']
            expected_cmd = ['/usr/bin/sbatch']
            expected_script = (
                '#!' + self.bash + '''

#SBATCH --export=ALL,'''
                '''FSLSUB_JOB_ID_VAR=SLURM_JOB_ID,'''
                '''FSLSUB_ARRAYTASKID_VAR=SLURM_ARRAY_TASK_ID,'''
                '''FSLSUB_ARRAYSTARTID_VAR=SLURM_ARRAY_TASK_MIN,'''
                '''FSLSUB_ARRAYENDID_VAR=SLURM_ARRAY_TASK_MAX,'''
                '''FSLSUB_ARRAYSTEPSIZE_VAR=SLURM_ARRAY_TASK_STEP,'''
                '''FSLSUB_ARRAYCOUNT_VAR=SLURM_ARRAY_TASK_COUNT,'''
                '''FSLSUB_NSLOTS=SLURM_NPROCS
#SBATCH --gres=gpu:k80:1
#SBATCH -o {0}.o%j
#SBATCH -e {0}.e%j
#SBATCH --job-name={1}
#SBATCH --chdir={6}
#SBATCH -p {2}
#SBATCH --parsable
#SBATCH --requeue
#SBATCH --account {3}
MODULEPATH=/usr/local/shellmodules:$MODULEPATH
module load mymodule
# Built by fsl_sub v.1.0.0 and fsl_sub_plugin_slurm v.2.0.0
# Command line: fsl_sub -q {2} {4}
# Submission time (H:M:S DD/MM/YYYY): {5}

{4}
'''.format(
                    os.path.join(logdir, job_name),
                    job_name, queue, project, ' '.join(cmd),
                    self.now.strftime("%H:%M:%S %d/%m/%Y"),
                    logdir)
            )
            mock_sprun.return_value = subprocess.CompletedProcess(
                expected_cmd, 0,
                stdout=qsub_out, stderr=None)
            with patch('fsl_sub.utils.sys.argv', ['fsl_sub', '-q', 'a.q', './acmd', 'arg1', 'arg2']):
                job_id = self.plugin.submit(
                    command=cmd,
                    job_name=job_name,
                    queue=queue,
                    project='Aproject',
                    coprocessor='cuda'
                )
                self.assertEqual(jid, job_id)
            mock_sprun.assert_called_once_with(
                expected_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                input=expected_script
            )

        mock_sprun.reset_mock()
        mock_sprun.reset_mock()
        with self.subTest("GPU without constraints"):
            w_conf = copy.deepcopy(self.config)
            w_conf['method_opts']['slurm']['projects'] = True
            w_conf['method_opts']['slurm']['add_module_paths'] = ['/usr/local/shellmodules']
            w_conf['copro_opts']['cuda']['classes'] = False
            self.mocks['fsl_sub_plugin_slurm.method_config'].return_value = w_conf['method_opts']['slurm']
            mock_cpconf.return_value = w_conf['copro_opts']['cuda']
            expected_cmd = ['/usr/bin/sbatch']
            expected_script = (
                '#!' + self.bash + '''

#SBATCH --export=ALL,'''
                '''FSLSUB_JOB_ID_VAR=SLURM_JOB_ID,'''
                '''FSLSUB_ARRAYTASKID_VAR=SLURM_ARRAY_TASK_ID,'''
                '''FSLSUB_ARRAYSTARTID_VAR=SLURM_ARRAY_TASK_MIN,'''
                '''FSLSUB_ARRAYENDID_VAR=SLURM_ARRAY_TASK_MAX,'''
                '''FSLSUB_ARRAYSTEPSIZE_VAR=SLURM_ARRAY_TASK_STEP,'''
                '''FSLSUB_ARRAYCOUNT_VAR=SLURM_ARRAY_TASK_COUNT,'''
                '''FSLSUB_NSLOTS=SLURM_NPROCS
#SBATCH --gres=gpu:1
#SBATCH -o {0}.o%j
#SBATCH -e {0}.e%j
#SBATCH --job-name={1}
#SBATCH --chdir={6}
#SBATCH -p {2}
#SBATCH --parsable
#SBATCH --requeue
#SBATCH --account {3}
MODULEPATH=/usr/local/shellmodules:$MODULEPATH
module load mymodule
# Built by fsl_sub v.1.0.0 and fsl_sub_plugin_slurm v.2.0.0
# Command line: fsl_sub -q {2} {4}
# Submission time (H:M:S DD/MM/YYYY): {5}

{4}
'''.format(
                    os.path.join(logdir, job_name),
                    job_name, queue, project, ' '.join(cmd),
                    self.now.strftime("%H:%M:%S %d/%m/%Y"),
                    logdir)
            )
            mock_sprun.return_value = subprocess.CompletedProcess(
                expected_cmd, 0,
                stdout=qsub_out, stderr=None)
            with patch('fsl_sub.utils.sys.argv', ['fsl_sub', '-q', 'a.q', './acmd', 'arg1', 'arg2']):
                job_id = self.plugin.submit(
                    command=cmd,
                    job_name=job_name,
                    queue=queue,
                    project='Aproject',
                    coprocessor='cuda'
                )
                self.assertEqual(jid, job_id)
            mock_sprun.assert_called_once_with(
                expected_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                input=expected_script
            )

        mock_sprun.reset_mock()
        with self.subTest("With GPU constraints"):
            w_conf = self.config
            w_conf['method_opts']['slurm']['projects'] = True
            w_conf['method_opts']['slurm']['add_module_paths'] = ['/usr/local/shellmodules']
            w_conf['copro_opts']['cuda']['set_visible'] = True
            w_conf['copro_opts']['cuda']['class_constraint'] = 'gpu_sku'
            self.mocks['fsl_sub_plugin_slurm.method_config'].return_value = w_conf['method_opts']['slurm']
            mock_cpconf.return_value = w_conf['copro_opts']['cuda']
            expected_cmd = ['/usr/bin/sbatch']
            expected_script = (
                '#!' + self.bash + '''

#SBATCH --export=ALL,'''
                '''FSLSUB_JOB_ID_VAR=SLURM_JOB_ID,'''
                '''FSLSUB_ARRAYTASKID_VAR=SLURM_ARRAY_TASK_ID,'''
                '''FSLSUB_ARRAYSTARTID_VAR=SLURM_ARRAY_TASK_MIN,'''
                '''FSLSUB_ARRAYENDID_VAR=SLURM_ARRAY_TASK_MAX,'''
                '''FSLSUB_ARRAYSTEPSIZE_VAR=SLURM_ARRAY_TASK_STEP,'''
                '''FSLSUB_ARRAYCOUNT_VAR=SLURM_ARRAY_TASK_COUNT,'''
                '''FSLSUB_NSLOTS=SLURM_NPROCS
#SBATCH --constraint="gpu_sku:k80|gpu_sku:p100"
#SBATCH --gres=gpu:1
#SBATCH -o {0}.o%j
#SBATCH -e {0}.e%j
#SBATCH --job-name={1}
#SBATCH --chdir={6}
#SBATCH -p {2}
#SBATCH --parsable
#SBATCH --requeue
#SBATCH --account {3}
MODULEPATH=/usr/local/shellmodules:$MODULEPATH
module load mymodule
# Built by fsl_sub v.1.0.0 and fsl_sub_plugin_slurm v.2.0.0
# Command line: fsl_sub -q {2} {4}
# Submission time (H:M:S DD/MM/YYYY): {5}

{4}
'''.format(
                    os.path.join(logdir, job_name),
                    job_name, queue, project, ' '.join(cmd),
                    self.now.strftime("%H:%M:%S %d/%m/%Y"),
                    logdir)
            )
            mock_sprun.return_value = subprocess.CompletedProcess(
                expected_cmd, 0,
                stdout=qsub_out, stderr=None)
            with patch('fsl_sub.utils.sys.argv', ['fsl_sub', '-q', 'a.q', './acmd', 'arg1', 'arg2']):
                job_id = self.plugin.submit(
                    command=cmd,
                    job_name=job_name,
                    queue=queue,
                    project='Aproject',
                    coprocessor='cuda'
                )
                self.assertEqual(jid, job_id)
            mock_sprun.assert_called_once_with(
                expected_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                input=expected_script
            )

    def test_submit_wrapper_set_vars(
            self, mock_sprun, mock_cpconf,
            mock_srbs, mock_qsub,
            mock_getcwd):
        job_name = 'test_job'
        queue = 'a.q'
        cmd = ['./acmd', 'arg1', 'arg2', ]
        logdir = os.getcwd()
        jid = 12345
        qsub_out = str(jid)
        w_conf = self.config
        w_conf['method_opts']['slurm']['use_jobscript'] = True
        w_conf['method_opts']['slurm']['copy_environment'] = False
        self.mocks['fsl_sub_plugin_slurm.method_config'].return_value = w_conf['method_opts']['slurm']
        mock_cpconf.return_value = w_conf['copro_opts']['cuda']

        expected_cmd = ['/usr/bin/sbatch']
        expected_script = (
            '''#!{0}

#SBATCH --export=FSLSUB_JOB_ID_VAR=SLURM_JOB_ID,\
FSLSUB_ARRAYTASKID_VAR=SLURM_ARRAY_TASK_ID,\
FSLSUB_ARRAYSTARTID_VAR=SLURM_ARRAY_TASK_MIN,\
FSLSUB_ARRAYENDID_VAR=SLURM_ARRAY_TASK_MAX,\
FSLSUB_ARRAYSTEPSIZE_VAR=SLURM_ARRAY_TASK_STEP,\
FSLSUB_ARRAYCOUNT_VAR=SLURM_ARRAY_TASK_COUNT,\
FSLSUB_NSLOTS=SLURM_NPROCS
#SBATCH -o {1}.o%j
#SBATCH -e {1}.e%j
#SBATCH --job-name={2}
#SBATCH --chdir={6}
#SBATCH -p {3}
#SBATCH --parsable
#SBATCH --requeue
module load mymodule
# Built by fsl_sub v.1.0.0 and fsl_sub_plugin_slurm v.2.0.0
# Command line: fsl_sub -q {3} {4}
# Submission time (H:M:S DD/MM/YYYY): {5}

{4}
'''.format(
                self.bash,
                os.path.join(logdir, job_name),
                job_name,
                queue, ' '.join(cmd),
                self.now.strftime("%H:%M:%S %d/%m/%Y"),
                logdir)
        )
        mock_sprun.return_value = subprocess.CompletedProcess(
            expected_cmd, 0,
            stdout=qsub_out, stderr=None)
        with patch('fsl_sub.utils.sys.argv', ['fsl_sub', '-q', 'a.q', './acmd', 'arg1', 'arg2']):
            self.assertEqual(
                jid,
                self.plugin.submit(
                    command=cmd,
                    job_name=job_name,
                    queue=queue
                )
            )
        mock_sprun.assert_called_once_with(
            expected_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            input=expected_script
        )
        mock_sprun.reset_mock()

    def test_submit_wrapper_set_complex_vars(
            self, mock_sprun, mock_cpconf,
            mock_srbs, mock_qsub,
            mock_getcwd):
        job_name = 'test_job'
        queue = 'a.q'
        cmd = ['./acmd', 'arg1', 'arg2', ]
        logdir = os.getcwd()
        jid = 12345
        qsub_out = str(jid)
        w_conf = self.config
        w_conf['method_opts']['slurm']['use_jobscript'] = True
        w_conf['method_opts']['slurm']['copy_environment'] = False
        self.mocks['fsl_sub_plugin_slurm.method_config'].return_value = w_conf['method_opts']['slurm']
        mock_cpconf.return_value = w_conf['copro_opts']['cuda']

        expected_cmd = ['/usr/bin/sbatch']
        expected_script = (
            '''#!{0}

#SBATCH --export=AVAR='1,2',BVAR='a b',FSLSUB_JOB_ID_VAR=SLURM_JOB_ID,\
FSLSUB_ARRAYTASKID_VAR=SLURM_ARRAY_TASK_ID,\
FSLSUB_ARRAYSTARTID_VAR=SLURM_ARRAY_TASK_MIN,\
FSLSUB_ARRAYENDID_VAR=SLURM_ARRAY_TASK_MAX,\
FSLSUB_ARRAYSTEPSIZE_VAR=SLURM_ARRAY_TASK_STEP,\
FSLSUB_ARRAYCOUNT_VAR=SLURM_ARRAY_TASK_COUNT,\
FSLSUB_NSLOTS=SLURM_NPROCS
#SBATCH -o {1}.o%j
#SBATCH -e {1}.e%j
#SBATCH --job-name={2}
#SBATCH --chdir={6}
#SBATCH -p {3}
#SBATCH --parsable
#SBATCH --requeue
module load mymodule
# Built by fsl_sub v.1.0.0 and fsl_sub_plugin_slurm v.2.0.0
# Command line: fsl_sub -q {3} {4}
# Submission time (H:M:S DD/MM/YYYY): {5}

{4}
'''.format(
                self.bash,
                os.path.join(logdir, job_name),
                job_name,
                queue, ' '.join(cmd),
                self.now.strftime("%H:%M:%S %d/%m/%Y"),
                logdir)
        )
        mock_sprun.return_value = subprocess.CompletedProcess(
            expected_cmd, 0,
            stdout=qsub_out, stderr=None)
        with patch('fsl_sub.utils.sys.argv', ['fsl_sub', '-q', 'a.q', './acmd', 'arg1', 'arg2']):
            self.assertEqual(
                jid,
                self.plugin.submit(
                    command=cmd,
                    job_name=job_name,
                    queue=queue,
                    export_vars=['AVAR=1,2', 'BVAR=a b']
                )
            )
        mock_sprun.assert_called_once_with(
            expected_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            input=expected_script
        )
        mock_sprun.reset_mock()

    def test_submit_wrapper_keep(
            self, mock_sprun, mock_cpconf,
            mock_srbs, mock_qsub,
            mock_getcwd):
        job_name = 'test_job'
        queue = 'a.q'
        cmd = ['./acmd', 'arg1', 'arg2', ]
        logdir = os.getcwd()
        jid = 12345
        qsub_out = str(jid)
        w_conf = self.config
        w_conf['method_opts']['slurm']['use_jobscript'] = True
        w_conf['method_opts']['slurm']['copy_environment'] = False
        self.mocks['fsl_sub_plugin_slurm.method_config'].return_value = w_conf['method_opts']['slurm']
        mock_cpconf.return_value = w_conf['copro_opts']['cuda']

        expected_cmd = ['/usr/bin/sbatch', self.ww.name]
        expected_script = [
            '#!' + self.bash,
            '',
            '#SBATCH --export=FSLSUB_JOB_ID_VAR=SLURM_JOB_ID,'
            'FSLSUB_ARRAYTASKID_VAR=SLURM_ARRAY_TASK_ID,'
            'FSLSUB_ARRAYSTARTID_VAR=SLURM_ARRAY_TASK_MIN,'
            'FSLSUB_ARRAYENDID_VAR=SLURM_ARRAY_TASK_MAX,'
            'FSLSUB_ARRAYSTEPSIZE_VAR=SLURM_ARRAY_TASK_STEP,'
            'FSLSUB_ARRAYCOUNT_VAR=SLURM_ARRAY_TASK_COUNT,'
            'FSLSUB_NSLOTS=SLURM_NPROCS',
            '#SBATCH -o {0}.o%j'.format(os.path.join(logdir, job_name)),
            '#SBATCH -e {0}.e%j'.format(os.path.join(logdir, job_name)),
            '#SBATCH --job-name=' + job_name,
            '#SBATCH --chdir=' + logdir,
            '#SBATCH -p ' + queue,
            '#SBATCH --parsable',
            '#SBATCH --requeue',
            'module load mymodule',
            '# Built by fsl_sub v.1.0.0 and fsl_sub_plugin_slurm v.2.0.0',
            '# Command line: fsl_sub -q {0} {1}'.format(queue, ' '.join(cmd)),
            '# Submission time (H:M:S DD/MM/YYYY): ' + self.now.strftime("%H:%M:%S %d/%m/%Y"),
            '',
            ' '.join(cmd),
            ''
        ]
        mock_sprun.return_value = subprocess.CompletedProcess(
            expected_cmd, 0,
            stdout=qsub_out, stderr=None)

        with patch('fsl_sub_plugin_slurm.os.rename') as mock_rename:
            with patch('fsl_sub.utils.sys.argv', ['fsl_sub', '-q', 'a.q', './acmd', 'arg1', 'arg2']):
                self.assertEqual(
                    jid,
                    self.plugin.submit(
                        command=cmd,
                        job_name=job_name,
                        queue=queue,
                        keep_jobscript=True
                    )
                )
        mock_sprun.assert_called_once_with(
            expected_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        mock_sprun.reset_mock()
        self.ww.seek(0)
        wrapper_lines = self.ww.read().splitlines()
        self.maxDiff = None
        self.assertListEqual(
            wrapper_lines,
            expected_script
        )
        mock_rename.assert_called_once_with(
            self.ww.name,
            os.path.join(
                os.getcwd(),
                '_'.join(('wrapper', str(jid) + '.sh'))
            )
        )


class TestQdel(unittest.TestCase):
    @patch('fsl_sub_plugin_slurm.which', autospec=True)
    @patch('fsl_sub_plugin_slurm.sp.run', autospec=True)
    def testqdel(self, mock_spr, mock_which):
        pid = 1234
        mock_which.return_value = '/usr/bin/scancel'
        mock_spr.return_value = subprocess.CompletedProcess(
            ['/usr/bin/cancel', str(pid)],
            0,
            'Job ' + str(pid) + ' deleted',
            ''
        )
        self.assertTupleEqual(
            fsl_sub_plugin_slurm.qdel(pid),
            ('Job ' + str(pid) + ' deleted', 0)
        )
        mock_spr.assert_called_once_with(
            ['/usr/bin/scancel', str(pid)],
            universal_newlines=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )


class TestJobStatus(unittest.TestCase):
    def setUp(self):
        self.QUEUED = 0
        self.RUNNING = 1
        self.FINISHED = 2
        self.FAILED = 3
        self.HELD = 4
        self.REQUEUED = 5
        self.RESTARTED = 6
        self.SUSPENDED = 7
        self.STARTING = 8

        self.REPORTING = [
            'Queued',
            'Running',
            'Finished',
            'Failed',
            'Held',
            'Requeued',
            'Restarted',
            'Suspended',
            'Starting'
        ]

        self.sacct_finished_out = (
            '''123456|myjob|2017-10-16T05:28:38|2017-10-16T05:29:24|2017-10-16T06:25:45|'''
            '''COMPLETED|0:0|''')
        self.sacct_finished_job = {
            'id': 123456,
            'name': 'myjob',
            'sub_time': datetime.datetime(2017, 10, 16, 5, 28, 38),
            'tasks': {
                1: {
                    'status': self.FINISHED,
                    'start_time': datetime.datetime(2017, 10, 16, 5, 29, 24),
                    'end_time': datetime.datetime(2017, 10, 16, 6, 25, 45),
                },
            },
        }
        self.sacct_failedbatch_out = (
            '''123456|feat|2020-10-19T12:00:49|2020-10-19T12:00:51|2020-10-19T12:15:07|FAILED|1:0
123456.batch|batch|2020-10-19T12:00:51|2020-10-19T12:00:51|2020-10-19T12:15:07|FAILED|1:0''')
        self.sacct_failedbatch_job = {
            'id': 123456,
            'name': 'feat',
            'sub_time': datetime.datetime(2020, 10, 19, 12, 00, 49),
            'tasks': {
                1: {
                    'status': self.FAILED,
                    'start_time': datetime.datetime(2020, 10, 19, 12, 00, 51),
                    'end_time': datetime.datetime(2020, 10, 19, 12, 15, 7),
                },
            },
        }
        self.expected_keys = [
            'id', 'name', 'sub_time', 'tasks',
        ]
        self.expected_keys.sort()

        self.task_expected_keys = [
            'status', 'start_time', 'end_time',
        ]
        self.task_expected_keys.sort()

        self.slurm_example_sacct = (
            '''1716106|acctest|2018-06-05T09:42:24|2018-06-05T09:42:24|'''
            '''2018-06-05T09:44:08|COMPLETED|0:0
1716106.batch|batch|2018-06-05T09:42:24|2018-06-05T09:42:24|'''
            '''2018-06-05T09:44:08|COMPLETED|0:0
''')

    @patch('fsl_sub_plugin_slurm._sacct_cmd', return_value='/usr/bin/sacct')
    def test_job_status(self, mock_qacct):
        self.maxDiff = None
        with patch('fsl_sub_plugin_slurm.sp.run', autospec=True) as mock_sprun:

            with self.subTest('No sacct'):
                mock_sprun.side_effect = FileNotFoundError
                self.assertRaises(
                    BadSubmission,
                    fsl_sub_plugin_slurm._get_sacct,
                    1716106)
            mock_sprun.reset_mock()
            mock_sprun.side_effect = None
            with self.subTest('No job'):
                mock_sprun.return_value = subprocess.CompletedProcess(
                    '/usr/bin/sacct',
                    stdout='',
                    stderr='',
                    returncode=0
                )
                self.assertRaises(
                    UnknownJobId,
                    fsl_sub_plugin_slurm._get_sacct,
                    1716106)
            mock_sprun.reset_mock()
            with self.subTest('Single job'):
                mock_sprun.return_value = subprocess.CompletedProcess(
                    '/usr/bin/sacct',
                    stdout=self.slurm_example_sacct,
                    returncode=0
                )
                self.assertDictEqual(
                    fsl_sub_plugin_slurm._get_sacct(1716106),
                    {
                        'id': 1716106,
                        'name': 'acctest',
                        'sub_time': datetime.datetime(
                            2018, 6, 5, 9, 42, 24),
                        'tasks': {
                            1: {
                                'start_time': datetime.datetime(
                                    2018, 6, 5, 9, 42, 24),
                                'end_time': datetime.datetime(
                                    2018, 6, 5, 9, 44, 8),
                                'status': fsl_sub.consts.FINISHED,
                            }
                        }
                    }
                )
        with self.subTest("Completed"):
            with patch('fsl_sub_plugin_slurm.sp.run', autospec=True) as mock_sprun:
                mock_sprun.return_value = subprocess.CompletedProcess(
                    ['sacct'], 0, self.sacct_finished_out, '')
                job_stat = fsl_sub_plugin_slurm.job_status(123456)
            output_keys = list(job_stat.keys())
            output_keys.sort()
            self.assertListEqual(output_keys, self.expected_keys)
            task_output_keys = list(job_stat['tasks'][1].keys())
            task_output_keys.sort()
            self.assertListEqual(task_output_keys, self.task_expected_keys)
            self.assertDictEqual(job_stat, self.sacct_finished_job)

        with self.subTest("Running"):
            with patch('fsl_sub_plugin_slurm.sp.run', autospec=True) as mock_sprun:
                mock_sprun.return_value = subprocess.CompletedProcess(
                    ['sacct'], 0, self.sacct_failedbatch_out, '')
                job_stat = fsl_sub_plugin_slurm.job_status(123456)
            output_keys = list(job_stat.keys())
            output_keys.sort()
            self.assertListEqual(output_keys, self.expected_keys)
            task_output_keys = list(job_stat['tasks'][1].keys())
            task_output_keys.sort()
            self.assertListEqual(task_output_keys, self.task_expected_keys)
            self.assertDictEqual(job_stat, self.sacct_failedbatch_job)


class TestQueueCapture(unittest.TestCase):
    def setUp(self):
        self.sinfo_f_one_host = '''os:centos7,
'''
        self.sinfo_f_two_host = (
            '''gpu,'''
            '''gpu_sku:P100,
            gpu,'''
            '''gpu_sku:V100,
''')
        self.sinfo_s = '''htc*\n'''
        self.sinfo_G_two_host = 'gpu:p100:4(S:0-1)\ngpu:v100:8(S:0-1)'
        self.sinfo_G_one_host = '(null)'
        self.sinfo_G_no_parens = 'gpu:gtx:2'
        self.sinfo_G_no_type = 'gpu:2(S:0-1)'
        self.sinfo_G_multiplier = 'gpu:2K(S:0-2023)'
        self.sinfo_G_list = 'gpu:p100:2(S:0-1),gpu:v100:2(S:0-1)'
        self.sinfo_O_one_host = '''8                  UNLIMITED           64000             1-00:00:00          htc-node1
'''
        self.sinfo_O_one_host_inf = '''8                  UNLIMITED           64000             infinite          htc-node1
'''
        self.sinfo_O_two_host = '''8                   UNLIMITED           384000               1-00:00:00          htc-gpu1
16                 UNLIMITED           512000               5-00:00:00          htc-gpu2
'''

    @patch('fsl_sub_plugin_slurm._sinfo_cmd', return_value='/usr/bin/sinfo')
    def test__get_queue_gres(self, mock_sinfo):
        with patch('fsl_sub_plugin_slurm.sp.run') as mock_spr:
            with self.subTest('No Type'):
                mock_spr.return_value = subprocess.CompletedProcess(
                    ['sinfo', '%G', ], 0, self.sinfo_G_no_type
                )
                gres = fsl_sub_plugin_slurm._get_queue_gres('htc')
                self.assertDictEqual(
                    gres,
                    {'gpu': [2]})
            with self.subTest("Multiplier"):
                mock_spr.return_value = subprocess.CompletedProcess(
                    ['sinfo', '%G', ], 0, self.sinfo_G_multiplier
                )
                gres = fsl_sub_plugin_slurm._get_queue_gres('htc')
                self.assertDictEqual(
                    gres,
                    {'gpu': [2048]})
            with self.subTest("No Parens"):
                mock_spr.return_value = subprocess.CompletedProcess(
                    ['sinfo', '%G', ], 0, self.sinfo_G_no_parens
                )
                gres = fsl_sub_plugin_slurm._get_queue_gres('htc')
                self.assertDictEqual(
                    gres,
                    {'gpu': [('gtx', 2), ]})
            with self.subTest("List"):
                mock_spr.return_value = subprocess.CompletedProcess(
                    ['sinfo', '%G', ], 0, self.sinfo_G_list
                )
                gres = fsl_sub_plugin_slurm._get_queue_gres('htc')
                self.assertDictEqual(
                    gres,
                    {'gpu': [('p100', 2), ('v100', 2), ]})
            with self.subTest("No GRES"):
                mock_spr.return_value = subprocess.CompletedProcess(
                    ['sinfo', '%G', ], 0, self.sinfo_G_one_host
                )
                gres = fsl_sub_plugin_slurm._get_queue_gres('htc')
                self.assertDictEqual(gres, defaultdict(list))
            with self.subTest('Two hosts'):
                mock_spr.return_value = subprocess.CompletedProcess(
                    ['sinfo', '%G', ], 0, self.sinfo_G_two_host
                )
                gres = fsl_sub_plugin_slurm._get_queue_gres('htc')
                self.assertDictEqual(gres, {'gpu': [('p100', 4, ), ('v100', 8, ), ], })

    @patch('fsl_sub_plugin_slurm._sinfo_cmd', return_value='/usr/bin/sinfo')
    def test__get_queue_info(self, mock_sinfo):
        with patch('fsl_sub_plugin_slurm.sp.run') as mock_spr:
            mock_spr.return_value = subprocess.CompletedProcess(
                ['sinfo', '%f', ], 0, self.sinfo_O_one_host
            )
            (qdef, comments) = fsl_sub_plugin_slurm._get_queue_info('htc')
            self.assertDictEqual(
                qdef,
                {
                    'cpus': 8,
                    'memory': 64,
                    'qname': 'htc',
                    'qtime': 1440,
                })
        with patch('fsl_sub_plugin_slurm.sp.run') as mock_spr:
            mock_spr.return_value = subprocess.CompletedProcess(
                ['sinfo', '%f', ], 0, self.sinfo_O_one_host_inf
            )
            (qdef, comments) = fsl_sub_plugin_slurm._get_queue_info('htc')
            self.assertDictEqual(
                qdef,
                {
                    'cpus': 8,
                    'memory': 64,
                    'qname': 'htc',
                    'qtime': 527039,
                })

    @patch('fsl_sub_plugin_slurm._sinfo_cmd', return_value='/usr/bin/sinfo')
    def test__get_queue_features(self, mock_sinfo):
        with patch('fsl_sub_plugin_slurm.sp.run') as mock_spr:
            mock_spr.return_value = subprocess.CompletedProcess(
                ['sinfo', '%f', ], 0, self.sinfo_f_one_host
            )
            features = fsl_sub_plugin_slurm._get_queue_features('htc')
            self.assertDictEqual(features, {'os': ['centos7', ], })
        with patch('fsl_sub_plugin_slurm.sp.run') as mock_spr:
            mock_spr.return_value = subprocess.CompletedProcess(
                ['sinfo', '%f', ], 0, self.sinfo_f_two_host
            )
            features = fsl_sub_plugin_slurm._get_queue_features('htc')
            self.assertDictEqual(features, {'gpu': [], 'gpu_sku': ['P100', 'V100'], })

    @patch('fsl_sub_plugin_slurm._sinfo_cmd', return_value='/usr/bin/sinfo')
    @patch('fsl_sub_plugin_slurm.method_config', return_value=conf_dict['method_opts']['slurm'])
    def test_build_queue_defs(self, mock_mconf, mock_sinfo):
        with patch('fsl_sub_plugin_slurm.sp.run') as mock_spr:
            mock_spr.side_effect = (
                subprocess.CompletedProcess(
                    ['sinfo', '-s', ], 0, self.sinfo_s
                ),
                subprocess.CompletedProcess(
                    ['sinfo', '%O', ], 0, self.sinfo_O_one_host
                ),
                subprocess.CompletedProcess(
                    ['sinfo', '%G', ], 0, self.sinfo_G_one_host
                ),
                subprocess.CompletedProcess(
                    ['sinfo', '%f', ], 0, self.sinfo_f_one_host
                )
            )
            qdefs = fsl_sub_plugin_slurm.build_queue_defs()
            yaml = YAML()
            yaml.width = 128
        expected_yaml = yaml.load('''queues:
  htc: # Queue name
  # default: true # Is this the default partition?
  # priority: 1 # Priority in group - higher wins
  # group: 1 # Group partitions with the same integer then order by priority
    time: 1440 # Maximum job run time in minutes
    max_slots: 8 # Maximum number of threads/slots on a queue
    max_size: 64 # Maximum RAM size of a job in {0}B
    slot_size: Null # Slot size is normally irrelevant on SLURM - set this to memory (in {0}B) per thread if required
'''.format(fsl_sub.consts.RAMUNITS))
        qd_str = io.StringIO()
        yaml.indent(mapping=2, sequence=4, offset=2)
        yaml.representer.add_representer(type(None), yaml_repr_none)
        yaml.dump(qdefs, qd_str)
        eq_str = io.StringIO()
        yaml.dump(expected_yaml, eq_str)
        self.maxDiff = None
        self.assertEqual(qd_str.getvalue(), eq_str.getvalue())
        with self.subTest("Two hosts"):
            with patch('fsl_sub_plugin_slurm.sp.run') as mock_spr:
                mock_spr.side_effect = (
                    subprocess.CompletedProcess(
                        ['sinfo', '-s', ], 0, self.sinfo_s
                    ),
                    subprocess.CompletedProcess(
                        ['sinfo', '%O', ], 0, self.sinfo_O_two_host
                    ),
                    subprocess.CompletedProcess(
                        ['sinfo', '%G', ], 0, self.sinfo_G_two_host
                    ),
                    subprocess.CompletedProcess(
                        ['sinfo', '%f', ], 0, self.sinfo_f_two_host
                    )
                )
                qdefs = fsl_sub_plugin_slurm.build_queue_defs()
                yaml = YAML()
                yaml.width = 128
            expected_yaml = yaml.load(
                '''queues:
  htc: # Queue name
  # Partition contains nodes with different numbers of CPUs
  # Partition contains nodes with different amounts of memory, consider switching on RAM nofitication
  # Partition contains nodes with differing maximum run times, consider switching on time notification
  # Partion has a GRES 'gpu' that might indicate the presence of GPUs, see below for possible configuration
  # coproc: cuda 'resource' would be 'gpu' and associated class resources:quantities would be:
  # p100:4
  # v100:8
  # Partition has features that look like GPU resources, these might be usable as constraints
  # If using constraints the coproc: cuda 'resource' could be gpu_sku and associated '''
                '''class 'resource's would be P100,V100
  # copros:
  #   cuda: # CUDA Co-processor available
  #     max_quantity: 8 # Maximum available per node
  #     classes: # List of classes (if classes supported)
  #       - p100
  #       - v100
  #     exclusive: False # Does this only run jobs requiring this co-processor?
  # default: true # Is this the default partition?
  # priority: 1 # Priority in group - higher wins
  # group: 1 # Group partitions with the same integer then order by priority
    time: 7200 # Maximum job run time in minutes
    max_slots: 16 # Maximum number of threads/slots on a queue
    max_size: 512 # Maximum RAM size of a job in {0}B
    slot_size: Null # Slot size is normally irrelevant on SLURM - set this to memory (in {0}B) per thread if required
'''.format(fsl_sub.consts.RAMUNITS))
            qd_str = io.StringIO()
            yaml.indent(mapping=2, sequence=4, offset=2)
            yaml.representer.add_representer(type(None), yaml_repr_none)
            yaml.dump(qdefs, qd_str)
            eq_str = io.StringIO()
            yaml.dump(expected_yaml, eq_str)
            self.maxDiff = None
            self.assertEqual(qd_str.getvalue(), eq_str.getvalue())


if __name__ == '__main__':
    unittest.main()
