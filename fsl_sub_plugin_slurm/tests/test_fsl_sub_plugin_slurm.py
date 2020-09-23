#!/usr/bin/env python
import copy
import datetime
import os
import subprocess
import tempfile
import unittest
import yaml
import fsl_sub_plugin_slurm

from unittest.mock import (patch, )

import fsl_sub.consts
from fsl_sub.exceptions import (
    BadSubmission,
    UnknownJobId
)

conf_dict = yaml.safe_load('''---
ram_units: G
method_opts:
    slurm:
        queues: True
        large_job_split_pe: threads
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
        map_ram: True
        set_time_limit: False
        thread_ram_divide: True
        job_priorities: False
        min_priority: 10000
        max_priority: 0
        array_holds: True
        array_limit: True
        architecture: False
        export_vars: []
        preserve_modules: True
        keep_jobscript: False
        preserve_modules: True
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
        slurm_constriants: True
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


class TestSlurmReporting(unittest.TestCase):
    @patch(
        'fsl_sub_plugin_slurm.sacct_cmd',
        autospec=True,
        return_value='/usr/bin/sacct')
    @patch(
        'fsl_sub_plugin_slurm.squeue_cmd',
        autospec=True,
        return_value='/usr/bin/squeue')
    @patch(
        'fsl_sub_plugin_slurm.sp.run',
        autospec=True
    )
    def test__get_squeue(self, mock_sprun, mock_squeue, mock_sacct):
        self.maxDiff = None
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
            slurm_example_sacct = (
                '''1716106|acctest|2018-06-05T09:42:24|2018-06-05T09:42:24|'''
                '''2018-06-05T09:44:08|00:00:00|00:00.004|COMPLETED|0:0|
1716106.batch|batch|2018-06-05T09:42:24|2018-06-05T09:42:24|'''
                '''2018-06-05T09:44:08|00:00:00|00:00.004|COMPLETED|0:0|202.15M
''')
            mock_sprun.return_value = subprocess.CompletedProcess(
                '/usr/bin/sacct',
                stdout=slurm_example_sacct,
                returncode=0
            )
            self.assertDictEqual(
                fsl_sub_plugin_slurm._get_sacct(1716106),
                {
                    'id': 1716106,
                    'script': None,
                    'arguments': None,
                    'submission_time': datetime.datetime(
                        2018, 6, 5, 9, 42, 24),
                    'parents': None,
                    'children': None,
                    'job_directory': None,
                    'name': 'acctest',
                    'tasks': {
                        1: {
                            'sub_time': datetime.datetime(
                                2018, 6, 5, 9, 42, 24),
                            'start_time': datetime.datetime(
                                2018, 6, 5, 9, 42, 24),
                            'end_time': datetime.datetime(
                                2018, 6, 5, 9, 44, 8),
                            'status': fsl_sub.consts.FINISHED,
                            'utime': 0.000,
                            'stime': 0.004,
                            'exit_status': 0,
                            'error_message': None,
                            'maxmemory': 202.15
                        }
                    }
                }
            )

    @patch(
        'fsl_sub_plugin_slurm.sacct_cmd',
        autospec=True,
        return_value='/usr/bin/sacct')
    @patch(
        'fsl_sub_plugin_slurm.sp.run',
        autospec=True
    )
    def test__get_sacct(self, mock_sprun, mock_saact):
        slurm_example_sacct = (
            '''1716106|acctest|2018-06-05T09:42:24|2018-06-05T09:42:24|'''
            '''2018-06-05T09:44:08|00:00:00|00:00.004|COMPLETED|0:0|
1716106.batch|batch|2018-06-05T09:42:24|2018-06-05T09:42:24|'''
            '''2018-06-05T09:44:08|00:00:00|00:00.004|COMPLETED|0:0|202.15M
''')
        self.maxDiff = None
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
                stdout=slurm_example_sacct,
                returncode=0
            )
            self.assertDictEqual(
                fsl_sub_plugin_slurm._get_sacct(1716106),
                {
                    'id': 1716106,
                    'script': None,
                    'arguments': None,
                    'submission_time': datetime.datetime(
                        2018, 6, 5, 9, 42, 24),
                    'parents': None,
                    'children': None,
                    'job_directory': None,
                    'name': 'acctest',
                    'tasks': {
                        1: {
                            'sub_time': datetime.datetime(
                                2018, 6, 5, 9, 42, 24),
                            'start_time': datetime.datetime(
                                2018, 6, 5, 9, 42, 24),
                            'end_time': datetime.datetime(
                                2018, 6, 5, 9, 44, 8),
                            'status': fsl_sub.consts.FINISHED,
                            'utime': 0.000,
                            'stime': 0.004,
                            'exit_status': 0,
                            'error_message': None,
                            'maxmemory': 202.15
                        }
                    }
                }
            )


class TestslurmFinders(unittest.TestCase):
    @patch('fsl_sub_plugin_slurm.qconf_cmd', autospec=True)
    def test_qtest(self, mock_qconf):
        bin_path = '/usr/bin/scontrol'
        mock_qconf.return_value = bin_path
        self.assertEqual(
            bin_path,
            fsl_sub_plugin_slurm.qtest()
        )
        mock_qconf.assert_called_once_with()

    @patch('fsl_sub_plugin_slurm.which', autospec=True)
    def test_qconf(self, mock_which):
        bin_path = '/usr/bin/scontrol'
        with self.subTest("Test 1"):
            mock_which.return_value = bin_path
            self.assertEqual(
                bin_path,
                fsl_sub_plugin_slurm.qconf_cmd()
            )
        mock_which.reset_mock()
        with self.subTest("Test 2"):
            mock_which.return_value = None
            self.assertRaises(
                fsl_sub_plugin_slurm.BadSubmission,
                fsl_sub_plugin_slurm.qconf_cmd
            )

    @patch('fsl_sub_plugin_slurm.which', autospec=True)
    def test_qstat(self, mock_which):
        bin_path = '/usr/bin/squeue'
        with self.subTest("Test 1"):
            mock_which.return_value = bin_path
            self.assertEqual(
                bin_path,
                fsl_sub_plugin_slurm.qstat_cmd()
            )
        mock_which.reset_mock()
        with self.subTest("Test 2"):
            mock_which.return_value = None
            self.assertRaises(
                fsl_sub_plugin_slurm.BadSubmission,
                fsl_sub_plugin_slurm.qstat_cmd
            )

    @patch('fsl_sub_plugin_slurm.which', autospec=True)
    def test_qsub(self, mock_which):
        bin_path = '/usr/bin/sbatch'
        with self.subTest("Test 1"):
            mock_which.return_value = bin_path
            self.assertEqual(
                bin_path,
                fsl_sub_plugin_slurm.qsub_cmd()
            )
        mock_which.reset_mock()
        with self.subTest("Test 2"):
            mock_which.return_value = None
            self.assertRaises(
                fsl_sub_plugin_slurm.BadSubmission,
                fsl_sub_plugin_slurm.qsub_cmd
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
    'fsl_sub_plugin_slurm.qsub_cmd',
    autospec=True, return_value='/usr/bin/sbatch'
)
@patch('fsl_sub_plugin_slurm.split_ram_by_slots', autospec=True)
@patch('fsl_sub_plugin_slurm.coprocessor_config', autospec=True)
@patch('fsl_sub_plugin_slurm.sp.run', autospec=True)
class TestSubmit(unittest.TestCase):
    def setUp(self):
        fsl_sub_plugin_slurm.fsl_sub.config.read_config.cache_clear()
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
            'fsl_sub_plugin_slurm.read_config': {'autospec': True, 'return_value': self.config, },
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
        for l in content:
            self.ww.write(l + '\n')
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
                    self.now.strftime("%H:%M:%S %d/%m/%Y"))
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
                    queue)
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
#SBATCH -p {6}
#SBATCH -w {5}
#SBATCH --parsable
#SBATCH --requeue
module load mymodule
# Built by fsl_sub v.1.0.0 and fsl_sub_plugin_slurm v.2.0.0
# Command line: fsl_sub -q {7} {3}
# Submission time (H:M:S DD/MM/YYYY): {4}

{3}
'''.format(
                    os.path.join(logdir, job_name),
                    job_name,
                    ','.join([q.split('@')[0] for q in queue]),
                    ' '.join(cmd),
                    self.now.strftime("%H:%M:%S %d/%m/%Y"),
                    'host1',
                    'a.q,b.q',
                    ','.join(queue)
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
                    ','.join(queue)
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
            w_conf = self.config
            w_conf['method_opts']['slurm']['projects'] = True
            self.mocks['fsl_sub_plugin_slurm.read_config'].return_value = w_conf
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
                    self.now.strftime("%H:%M:%S %d/%m/%Y"))
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
                    self.now.strftime("%H:%M:%S %d/%m/%Y"))
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
            w_conf = self.config
            w_conf['method_opts']['slurm']['projects'] = True
            w_conf['method_opts']['slurm']['add_module_paths'] = ['/usr/local/shellmodules']
            self.mocks['fsl_sub_plugin_slurm.read_config'].return_value = w_conf
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
                    self.now.strftime("%H:%M:%S %d/%m/%Y"))
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
        with self.subTest("With set GPU mask"):
            w_conf = self.config
            w_conf['method_opts']['slurm']['projects'] = True
            w_conf['method_opts']['slurm']['add_module_paths'] = ['/usr/local/shellmodules']
            w_conf['copro_opts']['cuda']['set_visible'] = True
            self.mocks['fsl_sub_plugin_slurm.read_config'].return_value = w_conf
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
#SBATCH --constraint="k80|p100"
#SBATCH --gres=gpu:1
#SBATCH -o {0}.o%j
#SBATCH -e {0}.e%j
#SBATCH --job-name={1}
#SBATCH -p {2}
#SBATCH --parsable
#SBATCH --requeue
#SBATCH --account {3}
MODULEPATH=/usr/local/shellmodules:$MODULEPATH
module load mymodule
# Built by fsl_sub v.1.0.0 and fsl_sub_plugin_slurm v.2.0.0
# Command line: fsl_sub -q {2} {4}
# Submission time (H:M:S DD/MM/YYYY): {5}

if [ -n "$SGE_HGR_gpu" ]
then
  if [ -z "$CUDA_VISIBLE_DEVICES" ]
  then
    export CUDA_VISIBLE_DEVICES=${{SGE_HGR_gpu// /,}}
  fi
  if [ -z "$GPU_DEVICE_ORDINAL" ]
  then
    export GPU_DEVICE_ORDINAL=${{SGE_HGR_gpu// /,}}
  fi
fi
{4}
'''.format(
                    os.path.join(logdir, job_name),
                    job_name, queue, project, ' '.join(cmd),
                    self.now.strftime("%H:%M:%S %d/%m/%Y"))
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
                        project='Aproject',
                        coprocessor='cuda'
                    )
                )
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
        self.mocks['fsl_sub_plugin_slurm.read_config'].return_value = w_conf
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
                self.now.strftime("%H:%M:%S %d/%m/%Y"))
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
        self.mocks['fsl_sub_plugin_slurm.read_config'].return_value = w_conf
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

#     def test_submit_requeueable(
#             self, mock_sprun, mock_cpconf,
#             mock_srbs, mock_mconf, mock_qsub,
#             mock_getcwd, mock_readconf):
#         job_name = 'test_job'
#         queue = 'a.q'
#         cmd = ['acmd', 'arg1', 'arg2', ]

#         jid = 12345
#         qsub_out = 'Your job ' + str(jid) + ' ("acmd") has been submitted'
#         with self.subTest("Univa"):
#             expected_cmd = [
#                 '/usr/bin/qsub',
#                 '-V',
#                 '-binding',
#                 'linear:1',
#                 '-N', 'test_job',
#                 '-cwd', '-q', 'a.q',
#                 '-shell', 'n',
#                 '-b', 'y',
#                 'acmd', 'arg1', 'arg2'
#             ]
#             mock_sprun.return_value = subprocess.CompletedProcess(
#                 expected_cmd, 0,
#                 stdout=qsub_out, stderr=None)
#             self.assertEqual(
#                 jid,
#                 self.plugin.submit(
#                     command=cmd,
#                     job_name=job_name,
#                     queue=queue,
#                     requeueable=False,
#                 )
#             )
#             mock_sprun.assert_called_once_with(
#                 expected_cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 universal_newlines=True
#             )

#     def test_submit_logdir(
#             self, mock_sprun, mock_cpconf,
#             mock_srbs, mock_mconf, mock_qsub,
#             mock_getcwd, mock_readconf):
#         job_name = 'test_job'
#         queue = 'a.q'
#         cmd = ['acmd', 'arg1', 'arg2', ]

#         jid = 12345
#         qsub_out = 'Your job ' + str(jid) + ' ("acmd") has been submitted'
#         with self.subTest("logdir"):
#             expected_cmd = [
#                 '/usr/bin/qsub',
#                 '-V',
#                 '-binding',
#                 'linear:1',
#                 '-o', '/tmp/alog',
#                 '-e', '/tmp/alog',
#                 '-N', 'test_job',
#                 '-cwd', '-q', 'a.q',
#                 '-r', 'y',
#                 '-shell', 'n',
#                 '-b', 'y',
#                 'acmd', 'arg1', 'arg2'
#             ]
#             mock_sprun.return_value = subprocess.CompletedProcess(
#                 expected_cmd, 0,
#                 stdout=qsub_out, stderr=None)
#             self.assertEqual(
#                 jid,
#                 self.plugin.submit(
#                     command=cmd,
#                     job_name=job_name,
#                     queue=queue,
#                     logdir="/tmp/alog"
#                 )
#             )
#             mock_sprun.assert_called_once_with(
#                 expected_cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 universal_newlines=True
#             )

#     def test_no_env_submit(
#             self, mock_sprun, mock_cpconf,
#             mock_srbs, mock_mconf, mock_qsub,
#             mock_getcwd, mock_readconf):
#         test_dict = dict(mconf_dict)
#         test_dict['copy_environment'] = False
#         mock_mconf.return_value = test_dict
#         job_name = 'test_job'
#         queue = 'a.q'
#         cmd = ['acmd', 'arg1', 'arg2', ]

#         jid = 12345
#         qsub_out = 'Your job ' + str(jid) + ' ("acmd") has been submitted'
#         expected_cmd = [
#             '/usr/bin/qsub',
#             '-binding',
#             'linear:1',
#             '-N', 'test_job',
#             '-cwd', '-q', 'a.q',
#             '-r', 'y',
#             '-shell', 'n',
#             '-b', 'y',
#             'acmd', 'arg1', 'arg2'
#         ]
#         mock_sprun.return_value = subprocess.CompletedProcess(
#             expected_cmd, 0,
#             stdout=qsub_out, stderr=None)
#         self.assertEqual(
#             jid,
#             self.plugin.submit(
#                 command=cmd,
#                 job_name=job_name,
#                 queue=queue,
#                 )
#         )
#         mock_sprun.assert_called_once_with(
#             expected_cmd,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             universal_newlines=True
#         )

#     def test_no_affinity_submit(
#             self, mock_sprun, mock_cpconf,
#             mock_srbs, mock_mconf, mock_qsub,
#             mock_getcwd, mock_readconf):
#         test_dict = dict(mconf_dict)
#         test_dict['affinity_type'] = None
#         mock_mconf.return_value = test_dict
#         job_name = 'test_job'
#         queue = 'a.q'
#         cmd = ['acmd', 'arg1', 'arg2', ]

#         jid = 12345
#         qsub_out = 'Your job ' + str(jid) + ' ("acmd") has been submitted'
#         expected_cmd = [
#             '/usr/bin/qsub',
#             '-V',
#             '-N', 'test_job',
#             '-cwd', '-q', 'a.q',
#             '-r', 'y',
#             '-shell', 'n',
#             '-b', 'y',
#             'acmd', 'arg1', 'arg2'
#         ]
#         mock_sprun.return_value = subprocess.CompletedProcess(
#             expected_cmd, 0,
#             stdout=qsub_out, stderr=None)
#         self.assertEqual(
#             jid,
#             self.plugin.submit(
#                 command=cmd,
#                 job_name=job_name,
#                 queue=queue,
#                 )
#         )
#         mock_sprun.assert_called_once_with(
#                 expected_cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 universal_newlines=True
#             )

#     def test_priority_submit(
#             self, mock_sprun, mock_cpconf,
#             mock_srbs, mock_mconf, mock_qsub,
#             mock_getcwd, mock_readconf):
#             job_name = 'test_job'
#             queue = 'a.q'
#             cmd = ['acmd', 'arg1', 'arg2', ]

#             jid = 12345
#             qsub_out = 'Your job ' + str(jid) + ' ("acmd") has been submitted'
#             with self.subTest("No priorities"):
#                 test_dict = dict(mconf_dict)
#                 test_dict['job_priorities'] = False
#                 mock_mconf.return_value = test_dict
#                 expected_cmd = [
#                     '/usr/bin/qsub',
#                     '-V',
#                     '-binding',
#                     'linear:1',
#                     '-N', 'test_job',
#                     '-cwd', '-q', 'a.q',
#                     '-r', 'y',
#                     '-shell', 'n',
#                     '-b', 'y',
#                     'acmd', 'arg1', 'arg2'
#                 ]
#                 mock_sprun.return_value = subprocess.CompletedProcess(
#                     expected_cmd, 0,
#                     stdout=qsub_out, stderr=None)
#                 self.assertEqual(
#                     jid,
#                     self.plugin.submit(
#                         command=cmd,
#                         job_name=job_name,
#                         queue=queue,
#                         priority=1000
#                         )
#                 )
#                 mock_sprun.assert_called_once_with(
#                     expected_cmd,
#                     stdout=subprocess.PIPE,
#                     stderr=subprocess.PIPE,
#                     universal_newlines=True
#                 )
#             mock_sprun.reset_mock()
#             mock_mconf.return_value = mconf_dict
#             with self.subTest("With priorities"):
#                 mock_mconf.return_value = mconf_dict
#                 expected_cmd = [
#                     '/usr/bin/qsub',
#                     '-V',
#                     '-binding',
#                     'linear:1',
#                     '-p', str(-1000),
#                     '-N', 'test_job',
#                     '-cwd', '-q', 'a.q',
#                     '-r', 'y',
#                     '-shell', 'n',
#                     '-b', 'y',
#                     'acmd', 'arg1', 'arg2'
#                 ]
#                 mock_sprun.return_value = subprocess.CompletedProcess(
#                     expected_cmd, 0,
#                     stdout=qsub_out, stderr=None)
#                 self.assertEqual(
#                     jid,
#                     self.plugin.submit(
#                         command=cmd,
#                         job_name=job_name,
#                         queue=queue,
#                         priority=-1000
#                         )
#                 )
#                 mock_sprun.assert_called_once_with(
#                     expected_cmd,
#                     stdout=subprocess.PIPE,
#                     stderr=subprocess.PIPE,
#                     universal_newlines=True
#                 )
#             mock_sprun.reset_mock()
#             mock_mconf.return_value = mconf_dict
#             with self.subTest("With priorities (limited)"):
#                 mock_mconf.return_value = mconf_dict
#                 expected_cmd = [
#                     '/usr/bin/qsub',
#                     '-V',
#                     '-binding',
#                     'linear:1',
#                     '-p', str(-1023),
#                     '-N', 'test_job',
#                     '-cwd', '-q', 'a.q',
#                     '-r', 'y',
#                     '-shell', 'n',
#                     '-b', 'y',
#                     'acmd', 'arg1', 'arg2'
#                 ]
#                 mock_sprun.return_value = subprocess.CompletedProcess(
#                     expected_cmd, 0,
#                     stdout=qsub_out, stderr=None)
#                 self.assertEqual(
#                     jid,
#                     self.plugin.submit(
#                         command=cmd,
#                         job_name=job_name,
#                         queue=queue,
#                         priority=-2000
#                         )
#                 )
#                 mock_sprun.assert_called_once_with(
#                     expected_cmd,
#                     stdout=subprocess.PIPE,
#                     stderr=subprocess.PIPE,
#                     universal_newlines=True
#                 )
#             mock_sprun.reset_mock()
#             mock_mconf.return_value = mconf_dict
#             with self.subTest("With priorities positive"):
#                 mock_mconf.return_value = mconf_dict
#                 expected_cmd = [
#                     '/usr/bin/qsub',
#                     '-V',
#                     '-binding',
#                     'linear:1',
#                     '-p', str(0),
#                     '-N', 'test_job',
#                     '-cwd', '-q', 'a.q',
#                     '-r', 'y',
#                     '-shell', 'n',
#                     '-b', 'y',
#                     'acmd', 'arg1', 'arg2'
#                 ]
#                 mock_sprun.return_value = subprocess.CompletedProcess(
#                     expected_cmd, 0,
#                     stdout=qsub_out, stderr=None)
#                 self.assertEqual(
#                     jid,
#                     self.plugin.submit(
#                         command=cmd,
#                         job_name=job_name,
#                         queue=queue,
#                         priority=2000
#                         )
#                 )
#                 mock_sprun.assert_called_once_with(
#                     expected_cmd,
#                     stdout=subprocess.PIPE,
#                     stderr=subprocess.PIPE,
#                     universal_newlines=True
#                 )

#     def test_resources_submit(
#             self, mock_sprun, mock_cpconf,
#             mock_srbs, mock_mconf, mock_qsub,
#             mock_getcwd, mock_readconf):
#             job_name = 'test_job'
#             queue = 'a.q'
#             cmd = ['acmd', 'arg1', 'arg2', ]

#             jid = 12345
#             qsub_out = 'Your job ' + str(jid) + ' ("acmd") has been submitted'
#             with self.subTest("With single resource"):
#                 expected_cmd = [
#                     '/usr/bin/qsub',
#                     '-V',
#                     '-binding',
#                     'linear:1',
#                     '-l', 'ramlimit=1000',
#                     '-N', 'test_job',
#                     '-cwd', '-q', 'a.q',
#                     '-r', 'y',
#                     '-shell', 'n',
#                     '-b', 'y',
#                     'acmd', 'arg1', 'arg2'
#                 ]
#                 mock_sprun.return_value = subprocess.CompletedProcess(
#                     expected_cmd, 0,
#                     stdout=qsub_out, stderr=None)
#                 self.assertEqual(
#                     jid,
#                     self.plugin.submit(
#                         command=cmd,
#                         job_name=job_name,
#                         queue=queue,
#                         resources='ramlimit=1000'
#                         )
#                 )
#                 mock_sprun.assert_called_once_with(
#                     expected_cmd,
#                     stdout=subprocess.PIPE,
#                     stderr=subprocess.PIPE,
#                     universal_newlines=True
#                 )
#             mock_sprun.reset_mock()
#             with self.subTest("With multiple resources"):
#                 expected_cmd = [
#                     '/usr/bin/qsub',
#                     '-V',
#                     '-binding',
#                     'linear:1',
#                     '-l', 'resource1=1,resource2=2',
#                     '-N', 'test_job',
#                     '-cwd', '-q', 'a.q',
#                     '-r', 'y',
#                     '-shell', 'n',
#                     '-b', 'y',
#                     'acmd', 'arg1', 'arg2'
#                 ]
#                 mock_sprun.return_value = subprocess.CompletedProcess(
#                     expected_cmd, 0,
#                     stdout=qsub_out, stderr=None)
#                 self.assertEqual(
#                     jid,
#                     self.plugin.submit(
#                         command=cmd,
#                         job_name=job_name,
#                         queue=queue,
#                         resources=['resource1=1', 'resource2=2', ]
#                         )
#                 )
#                 mock_sprun.assert_called_once_with(
#                     expected_cmd,
#                     stdout=subprocess.PIPE,
#                     stderr=subprocess.PIPE,
#                     universal_newlines=True
#                 )

#     def test_job_hold_submit(
#             self, mock_sprun, mock_cpconf,
#             mock_srbs, mock_mconf, mock_qsub,
#             mock_getcwd, mock_readconf):
#         job_name = 'test_job'
#         queue = 'a.q'
#         cmd = ['acmd', 'arg1', 'arg2', ]

#         jid = 12345
#         hjid = 12344
#         qsub_out = 'Your job ' + str(jid) + ' ("acmd") has been submitted'
#         with self.subTest("Basic string"):
#             expected_cmd = [
#                 '/usr/bin/qsub',
#                 '-V',
#                 '-binding',
#                 'linear:1',
#                 '-hold_jid', str(hjid),
#                 '-N', 'test_job',
#                 '-cwd', '-q', 'a.q',
#                 '-r', 'y',
#                 '-shell', 'n',
#                 '-b', 'y',
#                 'acmd', 'arg1', 'arg2'
#             ]
#             mock_sprun.return_value = subprocess.CompletedProcess(
#                 expected_cmd, 0,
#                 stdout=qsub_out, stderr=None)
#             self.assertEqual(
#                 jid,
#                 self.plugin.submit(
#                     command=cmd,
#                     job_name=job_name,
#                     queue=queue,
#                     jobhold=hjid
#                     )
#             )
#             mock_sprun.assert_called_once_with(
#                 expected_cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 universal_newlines=True
#             )
#         with self.subTest("List"):
#             mock_sprun.reset_mock()
#             hjid2 = 23456
#             expected_cmd = [
#                 '/usr/bin/qsub',
#                 '-V',
#                 '-binding',
#                 'linear:1',
#                 '-hold_jid', ",".join((str(hjid), str(hjid2))),
#                 '-N', 'test_job',
#                 '-cwd', '-q', 'a.q',
#                 '-r', 'y',
#                 '-shell', 'n',
#                 '-b', 'y',
#                 'acmd', 'arg1', 'arg2'
#             ]
#             mock_sprun.return_value = subprocess.CompletedProcess(
#                 expected_cmd, 0,
#                 stdout=qsub_out, stderr=None)
#             self.assertEqual(
#                 jid,
#                 self.plugin.submit(
#                     command=cmd,
#                     job_name=job_name,
#                     queue=queue,
#                     jobhold=[hjid, hjid2]
#                     )
#             )
#             mock_sprun.assert_called_once_with(
#                 expected_cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 universal_newlines=True
#             )
#         with self.subTest("List of strings"):
#             mock_sprun.reset_mock()
#             hjid2 = 23456
#             expected_cmd = [
#                 '/usr/bin/qsub',
#                 '-V',
#                 '-binding',
#                 'linear:1',
#                 '-hold_jid', ",".join((str(hjid), str(hjid2))),
#                 '-N', 'test_job',
#                 '-cwd', '-q', 'a.q',
#                 '-r', 'y',
#                 '-shell', 'n',
#                 '-b', 'y',
#                 'acmd', 'arg1', 'arg2'
#             ]
#             mock_sprun.return_value = subprocess.CompletedProcess(
#                 expected_cmd, 0,
#                 stdout=qsub_out, stderr=None)
#             self.assertEqual(
#                 jid,
#                 self.plugin.submit(
#                     command=cmd,
#                     job_name=job_name,
#                     queue=queue,
#                     jobhold=[str(hjid), str(hjid2)]
#                     )
#             )
#             mock_sprun.assert_called_once_with(
#                 expected_cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 universal_newlines=True
#             )
#         with self.subTest("Tuple"):
#             mock_sprun.reset_mock()
#             hjid2 = 23456
#             expected_cmd = [
#                 '/usr/bin/qsub',
#                 '-V',
#                 '-binding',
#                 'linear:1',
#                 '-hold_jid', ",".join((str(hjid), str(hjid2))),
#                 '-N', 'test_job',
#                 '-cwd', '-q', 'a.q',
#                 '-r', 'y',
#                 '-shell', 'n',
#                 '-b', 'y',
#                 'acmd', 'arg1', 'arg2'
#             ]
#             mock_sprun.return_value = subprocess.CompletedProcess(
#                 expected_cmd, 0,
#                 stdout=qsub_out, stderr=None)
#             self.assertEqual(
#                 jid,
#                 self.plugin.submit(
#                     command=cmd,
#                     job_name=job_name,
#                     queue=queue,
#                     jobhold=(hjid, hjid2)
#                     )
#             )
#             mock_sprun.assert_called_once_with(
#                 expected_cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 universal_newlines=True
#             )
#         with self.subTest("Native"):
#             mock_sprun.reset_mock()
#             native_hjid = "1234,2345"
#             expected_cmd = [
#                 '/usr/bin/qsub',
#                 '-V',
#                 '-binding',
#                 'linear:1',
#                 '-hold_jid', native_hjid,
#                 '-N', 'test_job',
#                 '-cwd', '-q', 'a.q',
#                 '-r', 'y',
#                 '-shell', 'n',
#                 '-b', 'y',
#                 'acmd', 'arg1', 'arg2'
#             ]
#             mock_sprun.return_value = subprocess.CompletedProcess(
#                 expected_cmd, 0,
#                 stdout=qsub_out, stderr=None)
#             self.assertEqual(
#                 jid,
#                 self.plugin.submit(
#                     command=cmd,
#                     job_name=job_name,
#                     queue=queue,
#                     jobhold=native_hjid,
#                     )
#             )
#             mock_sprun.assert_called_once_with(
#                 expected_cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 universal_newlines=True
#             )

#     def test_no_array_hold_submit(
#             self, mock_sprun, mock_cpconf,
#             mock_srbs, mock_mconf, mock_qsub,
#             mock_getcwd, mock_readconf):
#         job_name = 'test_job'
#         queue = 'a.q'
#         cmd = ['acmd', 'arg1', 'arg2', ]

#         self.assertRaises(
#             self.plugin.BadSubmission,
#             self.plugin.submit,
#             command=cmd,
#             job_name=job_name,
#             queue=queue,
#             array_hold=12345
#         )

#     def test_no_array_limit_submit(
#             self, mock_sprun, mock_cpconf,
#             mock_srbs, mock_mconf, mock_qsub,
#             mock_getcwd, mock_readconf):
#         job_name = 'test_job'
#         queue = 'a.q'
#         cmd = ['acmd', 'arg1', 'arg2', ]

#         self.assertRaises(
#             self.plugin.BadSubmission,
#             self.plugin.submit,
#             command=cmd,
#             job_name=job_name,
#             queue=queue,
#             array_limit=5
#         )

#     def test_jobram_submit(
#             self, mock_sprun, mock_cpconf,
#             mock_srbs, mock_mconf, mock_qsub,
#             mock_getcwd, mock_readconf):
#         job_name = 'test_job'
#         queue = 'a.q'
#         jid = 123456
#         jobram = 1024
#         cmd = ['acmd', 'arg1', 'arg2', ]

#         qsub_out = 'Your job ' + str(jid) + ' ("acmd") has been submitted'
#         with self.subTest('Basic submission'):
#             expected_cmd = [
#                 '/usr/bin/qsub',
#                 '-V',
#                 '-binding',
#                 'linear:1',
#                 '-l', "h_vmem={0}G,m_mem_free={0}G".format(jobram),
#                 '-N', 'test_job',
#                 '-cwd', '-q', 'a.q',
#                 '-r', 'y',
#                 '-shell', 'n',
#                 '-b', 'y',
#                 'acmd', 'arg1', 'arg2'
#             ]
#             mock_sprun.return_value = subprocess.CompletedProcess(
#                 expected_cmd, 0,
#                 stdout=qsub_out, stderr=None)
#             self.assertEqual(
#                 jid,
#                 self.plugin.submit(
#                     command=cmd,
#                     job_name=job_name,
#                     queue=queue,
#                     jobram=jobram
#                     )
#             )
#             mock_sprun.assert_called_once_with(
#                 expected_cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 universal_newlines=True
#             )
#         mock_sprun.reset_mock()
#         with self.subTest("Split on RAM"):
#             threads = 2
#             split_ram = jobram // threads
#             mock_srbs.return_value = split_ram
#             expected_cmd = [
#                 '/usr/bin/qsub',
#                 '-V',
#                 '-binding',
#                 'linear:2',
#                 '-l', "h_vmem={0}G,m_mem_free={0}G".format(split_ram),
#                 '-N', 'test_job',
#                 '-cwd', '-q', 'a.q',
#                 '-r', 'y',
#                 '-shell', 'n',
#                 '-b', 'y',
#                 'acmd', 'arg1', 'arg2'
#             ]
#             mock_sprun.return_value = subprocess.CompletedProcess(
#                 expected_cmd, 0,
#                 stdout=qsub_out, stderr=None)
#             self.assertEqual(
#                 jid,
#                 self.plugin.submit(
#                     command=cmd,
#                     job_name=job_name,
#                     queue=queue,
#                     jobram=jobram,
#                     threads=threads,
#                     ramsplit=True
#                     )
#             )
#             mock_sprun.assert_called_once_with(
#                 expected_cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 universal_newlines=True
#             )
#         mock_sprun.reset_mock()
#         ram_override = "2048"
#         with self.subTest('Not overriding memory request 1'):
#             expected_cmd = [
#                 '/usr/bin/qsub',
#                 '-V',
#                 '-binding',
#                 'linear:1',
#                 '-l', "m_mem_free={0}G".format(ram_override),
#                 '-l', "h_vmem={0}G".format(jobram),
#                 '-N', 'test_job',
#                 '-cwd', '-q', 'a.q',
#                 '-r', 'y',
#                 '-shell', 'n',
#                 '-b', 'y',
#                 'acmd', 'arg1', 'arg2'
#             ]
#             mock_sprun.return_value = subprocess.CompletedProcess(
#                 expected_cmd, 0,
#                 stdout=qsub_out, stderr=None)
#             self.assertEqual(
#                 jid,
#                 self.plugin.submit(
#                     command=cmd,
#                     job_name=job_name,
#                     queue=queue,
#                     jobram=jobram,
#                     resources=['m_mem_free=' + ram_override + 'G']
#                     )
#             )
#             mock_sprun.assert_called_once_with(
#                 expected_cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 universal_newlines=True
#             )
#         mock_sprun.reset_mock()
#         with self.subTest('Not overriding memory request 2'):
#             expected_cmd = [
#                 '/usr/bin/qsub',
#                 '-V',
#                 '-binding',
#                 'linear:1',
#                 '-l', "h_vmem={0}G".format(ram_override),
#                 '-l', "m_mem_free={0}G".format(jobram),
#                 '-N', 'test_job',
#                 '-cwd', '-q', 'a.q',
#                 '-r', 'y',
#                 '-shell', 'n',
#                 '-b', 'y',
#                 'acmd', 'arg1', 'arg2'
#             ]
#             mock_sprun.return_value = subprocess.CompletedProcess(
#                 expected_cmd, 0,
#                 stdout=qsub_out, stderr=None)
#             self.assertEqual(
#                 jid,
#                 self.plugin.submit(
#                     command=cmd,
#                     job_name=job_name,
#                     queue=queue,
#                     jobram=jobram,
#                     resources=['h_vmem=' + ram_override + 'G']
#                     )
#             )
#             mock_sprun.assert_called_once_with(
#                 expected_cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 universal_newlines=True
#             )
#         mock_sprun.reset_mock()
#         with self.subTest('Not overriding memory request 3'):
#             expected_cmd = [
#                 '/usr/bin/qsub',
#                 '-V',
#                 '-binding',
#                 'linear:1',
#                 '-l', "m_mem_free={0}G,h_vmem={0}G".format(ram_override),
#                 '-N', 'test_job',
#                 '-cwd', '-q', 'a.q',
#                 '-r', 'y',
#                 '-shell', 'n',
#                 '-b', 'y',
#                 'acmd', 'arg1', 'arg2'
#             ]
#             mock_sprun.return_value = subprocess.CompletedProcess(
#                 expected_cmd, 0,
#                 stdout=qsub_out, stderr=None)
#             self.assertEqual(
#                 jid,
#                 self.plugin.submit(
#                     command=cmd,
#                     job_name=job_name,
#                     queue=queue,
#                     jobram=jobram,
#                     resources=[
#                         'm_mem_free=' + ram_override + 'G',
#                         'h_vmem=' + ram_override + 'G',
#                     ]
#                     )
#             )
#             mock_sprun.assert_called_once_with(
#                 expected_cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 universal_newlines=True
#             )
#         mock_sprun.reset_mock()

#     def test_mail_support(
#             self, mock_sprun, mock_cpconf,
#             mock_srbs, mock_mconf, mock_qsub,
#             mock_getcwd, mock_readconf):
#         job_name = 'test_job'
#         queue = 'a.q'
#         jid = 123456
#         mailto = 'auser@adomain.com'
#         mail_on = 'e'
#         cmd = ['acmd', 'arg1', 'arg2', ]

#         qsub_out = 'Your job ' + str(jid) + ' ("acmd") has been submitted'
#         with self.subTest("Test mail settings"):
#             expected_cmd = [
#                 '/usr/bin/qsub',
#                 '-V',
#                 '-binding',
#                 'linear:1',
#                 '-M', mailto,
#                 '-m', mail_on,
#                 '-N', 'test_job',
#                 '-cwd', '-q', 'a.q',
#                 '-r', 'y',
#                 '-shell', 'n',
#                 '-b', 'y',
#                 'acmd', 'arg1', 'arg2'
#             ]
#             mock_sprun.return_value = subprocess.CompletedProcess(
#                 expected_cmd, 0,
#                 stdout=qsub_out, stderr=None)
#             self.assertEqual(
#                 jid,
#                 self.plugin.submit(
#                     command=cmd,
#                     job_name=job_name,
#                     queue=queue,
#                     mailto=mailto,
#                     mail_on=mail_on
#                     )
#             )
#             mock_sprun.assert_called_once_with(
#                 expected_cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 universal_newlines=True
#             )
#         mock_sprun.reset_mock()
#         with self.subTest("Test for auto set of mail mode"):
#             expected_cmd = [
#                 '/usr/bin/qsub',
#                 '-V',
#                 '-binding',
#                 'linear:1',
#                 '-M', mailto,
#                 '-m', mconf_dict['mail_mode'],
#                 '-N', 'test_job',
#                 '-cwd', '-q', 'a.q',
#                 '-r', 'y',
#                 '-shell', 'n',
#                 '-b', 'y',
#                 'acmd', 'arg1', 'arg2'
#             ]
#             mock_sprun.return_value = subprocess.CompletedProcess(
#                 expected_cmd, 0,
#                 stdout=qsub_out, stderr=None)
#             self.assertEqual(
#                 jid,
#                 self.plugin.submit(
#                     command=cmd,
#                     job_name=job_name,
#                     queue=queue,
#                     mailto=mailto,
#                     )
#             )
#             mock_sprun.assert_called_once_with(
#                 expected_cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 universal_newlines=True
#             )
#         mock_sprun.reset_mock()
#         with self.subTest("Test for multiple mail modes"):
#             expected_cmd = [
#                 '/usr/bin/qsub',
#                 '-V',
#                 '-binding',
#                 'linear:1',
#                 '-M', mailto,
#                 '-m', 'a,e,b',
#                 '-N', 'test_job',
#                 '-cwd', '-q', 'a.q',
#                 '-r', 'y',
#                 '-shell', 'n',
#                 '-b', 'y',
#                 'acmd', 'arg1', 'arg2'
#             ]
#             mock_sprun.return_value = subprocess.CompletedProcess(
#                 expected_cmd, 0,
#                 stdout=qsub_out, stderr=None)
#             self.assertEqual(
#                 jid,
#                 self.plugin.submit(
#                     command=cmd,
#                     job_name=job_name,
#                     queue=queue,
#                     mailto=mailto,
#                     mail_on='f'
#                     )
#             )
#             mock_sprun.assert_called_once_with(
#                 expected_cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 universal_newlines=True
#             )

#         with self.subTest("Test for bad input"):
#             mail_on = 't'
#             self.assertRaises(
#                 self.plugin.BadSubmission,
#                 self.plugin.submit,
#                 command=cmd,
#                 job_name=job_name,
#                 queue=queue,
#                 mailto=mailto,
#                 mail_on=mail_on
#             )

#     def test_coprocessor_submit(
#             self, mock_sprun, mock_cpconf,
#             mock_srbs, mock_mconf, mock_qsub,
#             mock_getcwd, mock_readconf):
#         job_name = 'test_job'
#         queue = 'a.q'
#         jid = 123456
#         cmd = ['acmd', 'arg1', 'arg2', ]
#         copro_type = 'cuda'
#         cp_opts = conf_dict['copro_opts'][copro_type]
#         mock_cpconf.return_value = cp_opts
#         gpuclass = 'P'
#         gputype = cp_opts['class_types'][gpuclass]['resource']
#         second_gtype = cp_opts['class_types']['K']['resource']
#         qsub_out = 'Your job ' + str(jid) + ' ("acmd") has been submitted'
#         with self.subTest("Test basic GPU"):
#             expected_cmd = [
#                 '/usr/bin/qsub',
#                 '-V',
#                 '-l', 'gputype=' + '|'.join((second_gtype, gputype)),
#                 '-l', 'gpu=1',
#                 '-N', 'test_job',
#                 '-cwd', '-q', 'a.q',
#                 '-r', 'y',
#                 '-shell', 'n',
#                 '-b', 'y',
#                 'acmd', 'arg1', 'arg2'
#             ]
#             mock_sprun.return_value = subprocess.CompletedProcess(
#                 expected_cmd, 0,
#                 stdout=qsub_out, stderr=None)
#             self.assertEqual(
#                 jid,
#                 self.plugin.submit(
#                     command=cmd,
#                     job_name=job_name,
#                     queue=queue,
#                     coprocessor=copro_type,
#                     )
#             )
#             mock_sprun.assert_called_once_with(
#                 expected_cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 universal_newlines=True
#             )
#         mock_sprun.reset_mock()
#         with self.subTest("Test specific class of GPU"):
#             gpuclass = 'K'
#             gputype = cp_opts['class_types'][gpuclass]['resource']
#             expected_cmd = [
#                 '/usr/bin/qsub',
#                 '-V',
#                 '-l', 'gputype=' + gputype,
#                 '-l', 'gpu=1',
#                 '-N', 'test_job',
#                 '-cwd', '-q', 'a.q',
#                 '-r', 'y',
#                 '-shell', 'n',
#                 '-b', 'y',
#                 'acmd', 'arg1', 'arg2'
#             ]
#             mock_sprun.return_value = subprocess.CompletedProcess(
#                 expected_cmd, 0,
#                 stdout=qsub_out, stderr=None)
#             self.assertEqual(
#                 jid,
#                 self.plugin.submit(
#                     command=cmd,
#                     job_name=job_name,
#                     queue=queue,
#                     coprocessor=copro_type,
#                     coprocessor_class_strict=True
#                     )
#             )
#             mock_sprun.assert_called_once_with(
#                 expected_cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 universal_newlines=True
#             )
#         mock_sprun.reset_mock()
#         with self.subTest("Test more capable classes of GPU"):
#             gpuclass = 'K'
#             gputype = cp_opts['class_types'][gpuclass]['resource']
#             second_gtype = cp_opts['class_types']['P']['resource']
#             expected_cmd = [
#                 '/usr/bin/qsub',
#                 '-V',
#                 '-l', 'gputype={0}|{1}'.format(gputype, second_gtype),
#                 '-l', 'gpu=1',
#                 '-N', 'test_job',
#                 '-cwd', '-q', 'a.q',
#                 '-r', 'y',
#                 '-shell', 'n',
#                 '-b', 'y',
#                 'acmd', 'arg1', 'arg2'
#             ]
#             mock_sprun.return_value = subprocess.CompletedProcess(
#                 expected_cmd, 0,
#                 stdout=qsub_out, stderr=None)
#             self.assertEqual(
#                 jid,
#                 self.plugin.submit(
#                     command=cmd,
#                     job_name=job_name,
#                     queue=queue,
#                     coprocessor=copro_type,
#                     )
#             )
#             mock_sprun.assert_called_once_with(
#                 expected_cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 universal_newlines=True
#             )
#         mock_sprun.reset_mock()
#         with self.subTest("Test more capable classes of GPU (configuration)"):
#             test_mconf = dict(mconf_dict)
#             copro_opts = dict(cp_opts)
#             copro_opts['include_more_capable'] = False
#             gpuclass = 'K'
#             gputype = cp_opts['class_types'][gpuclass]['resource']
#             mock_cpconf.return_value = copro_opts
#             expected_cmd = [
#                 '/usr/bin/qsub',
#                 '-V',
#                 '-l', 'gputype=' + gputype,
#                 '-l', 'gpu=1',
#                 '-N', 'test_job',
#                 '-cwd', '-q', 'a.q',
#                 '-r', 'y',
#                 '-shell', 'n',
#                 '-b', 'y',
#                 'acmd', 'arg1', 'arg2'
#             ]
#             mock_mconf.return_value = test_mconf
#             mock_sprun.return_value = subprocess.CompletedProcess(
#                 expected_cmd, 0,
#                 stdout=qsub_out, stderr=None)
#             self.assertEqual(
#                 jid,
#                 self.plugin.submit(
#                     command=cmd,
#                     job_name=job_name,
#                     queue=queue,
#                     coprocessor=copro_type,
#                     )
#             )
#             mock_sprun.assert_called_once_with(
#                 expected_cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 universal_newlines=True
#             )
#         mock_sprun.reset_mock()
#         mock_cpconf.return_value = cp_opts
#         with self.subTest("Test multi-GPU"):
#             multi_gpu = 2
#             gpuclass = cp_opts['default_class']
#             gputype = 'k80|p100'
#             expected_cmd = [
#                 '/usr/bin/qsub',
#                 '-V',
#                 '-l', 'gputype=' + gputype,
#                 '-l', 'gpu=' + str(multi_gpu),
#                 '-N', 'test_job',
#                 '-cwd', '-q', 'a.q',
#                 '-r', 'y',
#                 '-shell', 'n',
#                 '-b', 'y',
#                 'acmd', 'arg1', 'arg2'
#             ]
#             mock_sprun.return_value = subprocess.CompletedProcess(
#                 expected_cmd, 0,
#                 stdout=qsub_out, stderr=None)
#             self.assertEqual(
#                 jid,
#                 self.plugin.submit(
#                     command=cmd,
#                     job_name=job_name,
#                     queue=queue,
#                     coprocessor=copro_type,
#                     coprocessor_multi=2
#                     )
#             )
#             mock_sprun.assert_called_once_with(
#                 expected_cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 universal_newlines=True
#             )

#     @patch('fsl_sub_plugin_slurm.qconf_cmd', autospec=True)
#     def test_parallel_env_submit(
#             self, mock_qconf, mock_sprun, mock_cpconf,
#             mock_srbs, mock_mconf, mock_qsub,
#             mock_getcwd, mock_readconf):
#         mock_qconf.return_value = '/usr/bin/qconf'
#         job_name = 'test_job'
#         queue = 'a.q'
#         cmd = ['acmd', 'arg1', 'arg2', ]

#         jid = 12345
#         qsub_out = 'Your job ' + str(jid) + ' ("acmd") has been submitted'
#         with self.subTest("One thread"):
#             expected_cmd = [
#                 '/usr/bin/qsub',
#                 '-pe', 'openmp', '1', '-w', 'e',
#                 '-V',
#                 '-binding',
#                 'linear:1',
#                 '-N', 'test_job',
#                 '-cwd', '-q', 'a.q',
#                 '-r', 'y',
#                 '-shell', 'n',
#                 '-b', 'y',
#                 'acmd', 'arg1', 'arg2'
#             ]
#             mock_sprun.return_value = subprocess.CompletedProcess(
#                 expected_cmd, 0,
#                 stdout=qsub_out, stderr=None)
#             self.assertEqual(
#                 jid,
#                 self.plugin.submit(
#                     command=cmd,
#                     job_name=job_name,
#                     queue=queue,
#                     parallel_env='openmp',
#                 )
#             )
#             mock_sprun.assert_called_once_with(
#                 expected_cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 universal_newlines=True
#             )
#         mock_sprun.reset_mock()
#         with self.subTest("Two threads"):
#             expected_cmd = [
#                 '/usr/bin/qsub',
#                 '-pe', 'openmp', str(2), '-w', 'e',
#                 '-V',
#                 '-binding',
#                 'linear:2',
#                 '-N', 'test_job',
#                 '-cwd', '-q', 'a.q',
#                 '-r', 'y',
#                 '-shell', 'n',
#                 '-b', 'y',
#                 'acmd', 'arg1', 'arg2'
#             ]
#             mock_sprun.return_value = subprocess.CompletedProcess(
#                 expected_cmd, 0,
#                 stdout=qsub_out, stderr=None)
#             self.assertEqual(
#                 jid,
#                 self.plugin.submit(
#                     command=cmd,
#                     job_name=job_name,
#                     queue=queue,
#                     parallel_env='openmp',
#                     threads=2
#                 )
#             )
#             mock_sprun.assert_called_once_with(
#                 expected_cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 universal_newlines=True
#             )

#         with self.subTest("Bad PE"):
#             mock_check_pe.side_effect = self.plugin.BadSubmission
#             self.assertRaises(
#                 self.plugin.BadSubmission,
#                 self.plugin.submit,
#                 command=cmd,
#                 job_name=job_name,
#                 queue=queue,
#                 parallel_env='openmp',
#                 threads=2
#             )

#     def test_array_hold_on_non_array_submit(
#             self, mock_sprun, mock_cpconf,
#             mock_srbs, mock_mconf, mock_qsub,
#             mock_getcwd, mock_readconf):
#         job_name = 'test_job'
#         queue = 'a.q'
#         cmd = ['acmd', 'arg1', 'arg2', ]

#         hjid = 12344
#         self.assertRaises(
#             self.plugin.BadSubmission,
#             self.plugin.submit,
#             command=cmd,
#             job_name=job_name,
#             queue=queue,
#             array_hold=hjid
#         )

#     @patch('fsl_sub_plugin_slurm.os.remove', autospec=True)
#     def test_array_submit(
#             self, mock_osr, mock_sprun, mock_cpconf,
#             mock_srbs, mock_mconf, mock_qsub,
#             mock_getcwd, mock_readconf):
#         job_name = 'test_job'
#         queue = 'a.q'
#         job_file = '''
# acmd 1 2 3
# acmd 4 5 6
# acmd 6 7 8
# '''
#         job_file_name = 'll_job'
#         tmp_file = 'atmpfile'
#         jid = 12344
#         qsub_out = 'Your job ' + str(jid) + ' ("test_job") has been submitted'
#         expected_cmd = [
#             '/usr/bin/qsub',
#             '-V',
#             '-binding',
#             'linear:1',
#             '-N', 'test_job',
#             '-cwd', '-q', 'a.q',
#             '-r', 'y',
#             '-t', "1-4",
#             tmp_file
#         ]
#         mock_tmpfile = mock_ntf.return_value.__enter__.return_value
#         mock_tmpfile.name = tmp_file
#         mock_write = mock_tmpfile.write
#         with patch(
#                 'fsl_sub_plugin_slurm.open',
#                 new_callable=mock_open, read_data=job_file) as m:
#             m.return_value.__iter__.return_value = job_file.splitlines()
#             mock_sprun.return_value = subprocess.CompletedProcess(
#                 expected_cmd, 0,
#                 stdout=qsub_out, stderr=None)
#             self.assertEqual(
#                 jid,
#                 self.plugin.submit(
#                     command=job_file_name,
#                     job_name=job_name,
#                     queue=queue,
#                     array_task=True
#                 )
#             )
#             mock_sprun.assert_called_once_with(
#                 expected_cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 universal_newlines=True
#             )
#             mock_ntf.assert_called_once_with(
#                 delete=False, mode='wt'
#             )
#             mock_write.assert_called_once_with(
#                 '''#!/bin/bash

# #$ -S /bin/bash

# the_command=$(sed -n -e "${{slurm_TASK_ID}}p" {0})

# exec /bin/bash -c "$the_command"
# '''.format(job_file_name)
#             )
#             mock_osr.assert_called_once_with(tmp_file)

#     @patch('fsl_sub_plugin_slurm.os.remove', autospec=True)
#     def test_array_submit_fails(
#             self, mock_osr, mock_sprun, mock_cpconf,
#             mock_srbs, mock_mconf, mock_qsub,
#             mock_getcwd, mock_readconf):
#         job_name = 'test_job'
#         queue = 'a.q'
#         job_file = '''
# acmd 1 2 3
# acmd 4 5 6
# acmd 6 7 8
# '''
#         job_file_name = 'll_job'
#         tmp_file = 'atmpfile'
#         jid = 12344
#         qsub_out = 'Your job ' + str(jid) + ' ("test_job") has been submitted'
#         expected_cmd = [
#             '/usr/bin/qsub',
#             '-V',
#             '-binding',
#             'linear:1',
#             '-N', 'test_job',
#             '-cwd', '-q', 'a.q',
#             '-r', 'y',
#             '-t', "1-4",
#             tmp_file
#         ]
#         mock_tmpfile = mock_ntf.return_value.__enter__.return_value
#         mock_tmpfile.name = tmp_file
#         mock_write = mock_tmpfile.write
#         with patch(
#                 'fsl_sub_plugin_slurm.open',
#                 new_callable=mock_open, read_data=job_file) as m:
#             m.return_value.__iter__.return_value = job_file.splitlines()
#             mock_sprun.return_value = subprocess.CompletedProcess(
#                 expected_cmd, 1, stdout=qsub_out, stderr="Bad job")
#             self.assertRaises(
#                 self.plugin.BadSubmission,
#                 self.plugin.submit,
#                 command=job_file_name,
#                 job_name=job_name,
#                 queue=queue,
#                 array_task=True
#             )
#             mock_sprun.assert_called_once_with(
#                 expected_cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 universal_newlines=True
#             )
#             mock_ntf.assert_called_once_with(
#                 delete=False, mode='wt'
#             )
#             mock_write.assert_called_once_with(
#                 '''#!/bin/bash

# #$ -S /bin/bash

# the_command=$(sed -n -e "${{slurm_TASK_ID}}p" {0})

# exec /bin/bash -c "$the_command"
# '''.format(job_file_name)
#             )
#             mock_osr.assert_called_once_with(tmp_file)

#     @patch('fsl_sub_plugin_slurm.os.remove', autospec=True)
#     def test_array_submit_specifier(
#             self, mock_osr, mock_sprun, mock_cpconf,
#             mock_srbs, mock_mconf, mock_qsub,
#             mock_getcwd, mock_readconf):
#         job_name = 'test_job'
#         queue = 'a.q'
#         job_file_name = 'll_job'
#         jid = 12344
#         qsub_out = 'Your job ' + str(jid) + ' ("test_job") has been submitted'
#         expected_cmd = [
#             '/usr/bin/qsub',
#             '-V',
#             '-binding',
#             'linear:1',
#             '-N', 'test_job',
#             '-cwd', '-q', 'a.q',
#             '-r', 'y',
#             '-t', "1-8:2",
#             '-shell', 'n', '-b', 'y',
#             job_file_name
#         ]
#         mock_sprun.return_value = subprocess.CompletedProcess(
#                 expected_cmd, 0,
#                 stdout=qsub_out, stderr=None)
#         self.assertEqual(
#             jid,
#             self.plugin.submit(
#                 command=job_file_name,
#                 job_name=job_name,
#                 queue=queue,
#                 array_task=True,
#                 array_specifier='1-8:2'
#             )
#         )
#         mock_sprun.assert_called_once_with(
#             expected_cmd,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             universal_newlines=True
#         )

#     @patch('fsl_sub_plugin_slurm.os.remove', autospec=True)
#     def test_array_limit_submit(
#             self, mock_osr, mock_sprun, mock_cpconf,
#             mock_srbs, mock_mconf, mock_qsub,
#             mock_getcwd, mock_readconf):
#         job_name = 'test_job'
#         queue = 'a.q'
#         job_file = '''
# acmd 1 2 3
# acmd 4 5 6
# acmd 6 7 8
# '''
#         job_file_name = 'll_job'
#         tmp_file = 'atmpfile'
#         jid = 12344
#         limit = 2
#         qsub_out = 'Your job ' + str(jid) + ' ("test_job") has been submitted'
#         expected_cmd = [
#             '/usr/bin/qsub',
#             '-V',
#             '-binding',
#             'linear:1',
#             '-tc', str(limit),
#             '-N', 'test_job',
#             '-cwd', '-q', 'a.q',
#             '-r', 'y',
#             '-t', "1-4",
#             tmp_file
#         ]
#         mock_tmpfile = mock_ntf.return_value.__enter__.return_value
#         mock_tmpfile.name = tmp_file
#         mock_write = mock_tmpfile.write
#         with patch(
#                 'fsl_sub_plugin_slurm.open',
#                 new_callable=mock_open, read_data=job_file) as m:
#             m.return_value.__iter__.return_value = job_file.splitlines()
#             mock_sprun.return_value = subprocess.CompletedProcess(
#                 expected_cmd, 0,
#                 stdout=qsub_out, stderr=None)
#             self.assertEqual(
#                 jid,
#                 self.plugin.submit(
#                     command=job_file_name,
#                     job_name=job_name,
#                     queue=queue,
#                     array_task=True,
#                     array_limit=limit
#                 )
#             )
#             mock_sprun.assert_called_once_with(
#                 expected_cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 universal_newlines=True
#             )
#             mock_ntf.assert_called_once_with(
#                 delete=False, mode='wt'
#             )
#             mock_write.assert_called_once_with(
#                 '''#!/bin/bash

# #$ -S /bin/bash

# the_command=$(sed -n -e "${{slurm_TASK_ID}}p" {0})

# exec /bin/bash -c "$the_command"
# '''.format(job_file_name)
#             )
#             mock_osr.assert_called_once_with(tmp_file)

#     @patch('fsl_sub_plugin_slurm.os.remove', autospec=True)
#     def test_array_limit_disabled_submit(
#             self, mock_osr, mock_sprun, mock_cpconf,
#             mock_srbs, mock_mconf, mock_qsub,
#             mock_getcwd, mock_readconf):
#         job_name = 'test_job'
#         queue = 'a.q'
#         job_file = '''
# acmd 1 2 3
# acmd 4 5 6
# acmd 6 7 8
# '''
#         job_file_name = 'll_job'
#         tmp_file = 'atmpfile'
#         jid = 12344
#         qsub_out = 'Your job ' + str(jid) + ' ("test_job") has been submitted'
#         test_mconf = dict(mconf_dict)
#         test_mconf['array_limit'] = False
#         mock_mconf.return_value = test_mconf
#         expected_cmd = [
#             '/usr/bin/qsub',
#             '-V',
#             '-binding',
#             'linear:1',
#             '-N', 'test_job',
#             '-cwd', '-q', 'a.q',
#             '-r', 'y',
#             '-t', "1-4",
#             tmp_file
#         ]
#         mock_tmpfile = mock_ntf.return_value.__enter__.return_value
#         mock_tmpfile.name = tmp_file
#         mock_write = mock_tmpfile.write
#         with patch(
#                 'fsl_sub_plugin_slurm.open',
#                 new_callable=mock_open, read_data=job_file) as m:
#             m.return_value.__iter__.return_value = job_file.splitlines()
#             mock_sprun.return_value = subprocess.CompletedProcess(
#                 expected_cmd, 0,
#                 stdout=qsub_out, stderr=None)
#             self.assertEqual(
#                 jid,
#                 self.plugin.submit(
#                     command=job_file_name,
#                     job_name=job_name,
#                     queue=queue,
#                     array_task=True,
#                     array_limit=2
#                 )
#             )
#             mock_sprun.assert_called_once_with(
#                 expected_cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 universal_newlines=True
#             )
#             mock_ntf.assert_called_once_with(
#                 delete=False, mode='wt'
#             )
#             mock_write.assert_called_once_with(
#                 '''#!/bin/bash

# #$ -S /bin/bash

# the_command=$(sed -n -e "${{slurm_TASK_ID}}p" {0})

# exec /bin/bash -c "$the_command"
# '''.format(job_file_name)
#             )
#             mock_osr.assert_called_once_with(tmp_file)

#     @patch('fsl_sub_plugin_slurm.os.remove', autospec=True)
#     def test_array_hold_submit(
#             self, mock_osr, mock_sprun, mock_cpconf,
#             mock_srbs, mock_mconf, mock_qsub,
#             mock_getcwd, mock_readconf):
#         job_name = 'test_job'
#         queue = 'a.q'
#         job_file = '''
# acmd 1 2 3
# acmd 4 5 6
# acmd 6 7 8
# '''
#         job_file_name = 'll_job'
#         tmp_file = 'atmpfile'
#         jid = 12344
#         hold_jid = 12343
#         qsub_out = 'Your job ' + str(jid) + ' ("test_job") has been submitted'
#         expected_cmd = [
#             '/usr/bin/qsub',
#             '-V',
#             '-binding',
#             'linear:1',
#             '-hold_jid_ad', str(hold_jid),
#             '-N', 'test_job',
#             '-cwd', '-q', 'a.q',
#             '-r', 'y',
#             '-t', "1-4",
#             tmp_file
#         ]
#         mock_tmpfile = mock_ntf.return_value.__enter__.return_value
#         mock_tmpfile.name = tmp_file
#         mock_write = mock_tmpfile.write
#         with patch(
#                 'fsl_sub_plugin_slurm.open',
#                 new_callable=mock_open, read_data=job_file) as m:
#             m.return_value.__iter__.return_value = job_file.splitlines()
#             mock_sprun.return_value = subprocess.CompletedProcess(
#                 expected_cmd, 0,
#                 stdout=qsub_out, stderr=None)
#             self.assertEqual(
#                 jid,
#                 self.plugin.submit(
#                     command=job_file_name,
#                     job_name=job_name,
#                     queue=queue,
#                     array_task=True,
#                     array_hold=hold_jid
#                 )
#             )
#             mock_sprun.assert_called_once_with(
#                 expected_cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 universal_newlines=True
#             )
#             mock_ntf.assert_called_once_with(
#                 delete=False, mode='wt'
#             )
#             mock_write.assert_called_once_with(
#                 '''#!/bin/bash

# #$ -S /bin/bash

# the_command=$(sed -n -e "${{slurm_TASK_ID}}p" {0})

# exec /bin/bash -c "$the_command"
# '''.format(job_file_name)
#             )
#             mock_osr.assert_called_once_with(tmp_file)

#     @patch('fsl_sub_plugin_slurm.os.remove', autospec=True)
#     def test_array_hold_disabled_submit(
#             self, mock_osr, mock_sprun, mock_cpconf,
#             mock_srbs, mock_mconf, mock_qsub,
#             mock_getcwd, mock_readconf):
#         job_name = 'test_job'
#         queue = 'a.q'
#         job_file = '''
# acmd 1 2 3
# acmd 4 5 6
# acmd 6 7 8
# '''
#         job_file_name = 'll_job'
#         tmp_file = 'atmpfile'
#         jid = 12344
#         hold_jid = 12343
#         qsub_out = 'Your job ' + str(jid) + ' ("test_job") has been submitted'
#         test_mconf = dict(mconf_dict)
#         test_mconf['array_holds'] = False
#         mock_mconf.return_value = test_mconf
#         expected_cmd = [
#             '/usr/bin/qsub',
#             '-V',
#             '-binding',
#             'linear:1',
#             '-hold_jid', str(hold_jid),
#             '-N', 'test_job',
#             '-cwd', '-q', 'a.q',
#             '-r', 'y',
#             '-t', "1-4",
#             tmp_file
#         ]
#         mock_tmpfile = mock_ntf.return_value.__enter__.return_value
#         mock_tmpfile.name = tmp_file
#         mock_write = mock_tmpfile.write
#         with patch(
#                 'fsl_sub_plugin_slurm.open',
#                 new_callable=mock_open, read_data=job_file) as m:
#             m.return_value.__iter__.return_value = job_file.splitlines()
#             mock_sprun.return_value = subprocess.CompletedProcess(
#                 expected_cmd, 0,
#                 stdout=qsub_out, stderr=None)
#             self.assertEqual(
#                 jid,
#                 self.plugin.submit(
#                     command=job_file_name,
#                     job_name=job_name,
#                     queue=queue,
#                     array_task=True,
#                     array_hold=hold_jid
#                 )
#             )
#             mock_sprun.assert_called_once_with(
#                 expected_cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 universal_newlines=True
#             )
#             mock_ntf.assert_called_once_with(
#                 delete=False, mode='wt'
#             )
#             mock_write.assert_called_once_with(
#                 '''#!/bin/bash

# #$ -S /bin/bash

# the_command=$(sed -n -e "${{slurm_TASK_ID}}p" {0})

# exec /bin/bash -c "$the_command"
# '''.format(job_file_name)
#             )
#             mock_osr.assert_called_once_with(tmp_file)


if __name__ == '__main__':
    unittest.main()
