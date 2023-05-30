# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
# Obtained from: https://github.com/lhoyer/HRDA
# Modifications: Add startup test

import argparse
import json
import logging
import os
import subprocess
import uuid
from datetime import datetime
import time

import torch
import mmcv
from experiments import generate_experiment_cfgs
from mmcv import Config, get_git_hash
from tools import train


def run_command(command):
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    for line in iter(p.stdout.readline, b''):
        print(line.decode('utf-8'), end='')


def rsync(src, dst):
    rsync_cmd = f'rsync -a {src} {dst}'
    print(rsync_cmd)
    run_command(rsync_cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--exp',
        type=int,
        default=None,
        help='Experiment id as defined in experiment.py',
    )
    group.add_argument(
        '--config',
        default=None,
        help='Path to config file',
    )
    parser.add_argument(
        '--machine', type=str, choices=['local'], default='local')
    parser.add_argument(
        '--local_rank', type=int, default=0)
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Path to checkpoint file',
    )
    parser.add_argument(
        '--seed-to-resume-from',
        type=int,
        default=0,
        help='Seed ID of experiment that needs resuming',
    )
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--startup-test', action='store_true')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    assert (args.config is None) != (args.exp is None), \
        'Either config or exp has to be defined.'

    GEN_CONFIG_DIR = 'configs/generated/'
    JOB_DIR = 'jobs'
    cfgs, config_files = [], []

    # Training with Predefined Config
    if args.config is not None:
        cfg = Config.fromfile(args.config)
        # Specify Name and Work Directory
        exp_name = f'{args.machine}-{cfg["exp"]}'
        unique_name = f'{datetime.now().strftime("%y%m%d_%H%M")}_' \
                      f'{cfg["name"]}_{str(uuid.uuid4())[:5]}'
        child_cfg = {
            '_base_': args.config.replace('configs', '../..'),
            'name': unique_name,
            'work_dir': os.path.join('work_dirs', exp_name, unique_name),
            'git_rev': get_git_hash()
        }
        cfg_out_file = f"{GEN_CONFIG_DIR}/{exp_name}/{child_cfg['name']}.json"
        os.makedirs(os.path.dirname(cfg_out_file), exist_ok=True)
        assert not os.path.isfile(cfg_out_file)
        with open(cfg_out_file, 'w') as of:
            json.dump(child_cfg, of, indent=4)
        config_files.append(cfg_out_file)
        cfgs.append(cfg)

    # Training with Generated Configs from experiments.py
    if args.exp is not None:
        exp_name = f'{args.machine}-exp{args.exp}'
        if args.startup_test:
            exp_name += '-startup'
        cfgs = generate_experiment_cfgs(args.exp)
        # Generate Configs
        for i, cfg in enumerate(cfgs):
            if args.debug:
                cfg.setdefault('log_config', {})['interval'] = 10
                cfg['evaluation'] = dict(interval=200, metric='mIoU')
                if 'dacs' in cfg['name']:
                    cfg.setdefault('uda', {})['debug_img_interval'] = 10
                    # cfg.setdefault('uda', {})['print_grad_magnitude'] = True
            if args.startup_test:
                cfg['log_level'] = logging.ERROR
                cfg['runner'] = dict(type='IterBasedRunner', max_iters=2)
                cfg['evaluation']['interval'] = 100
                cfg['checkpoint_config'] = dict(
                    by_epoch=False, interval=100, save_last=False)
            # Generate Config File
            # cfg['runner'] = dict(type='IterBasedRunner', max_iters=2)
            # cfg['evaluation'] = dict(interval=2, metric='mIoU', distributed_eval=True, pre_eval=True)
            # cfg.setdefault('log_config', {})['interval'] = 1
            # cfg['checkpoint_config'] = dict(by_epoch=False, interval=1, max_keep_ckpts=1)
            if 'SLURM_ARRAY_TASK_ID' not in os.environ and ('LSB_JOBINDEX' not in os.environ or int(os.environ['LSB_JOBINDEX']) == 0) and i == args.seed_to_resume_from:
                cfg['resume_from'] = args.resume_from
                cfg['first_run'] = True
            # In case of job array, only run the configuration with number that corresponds to the task ID.
            if 'SLURM_ARRAY_TASK_ID' in os.environ:
                if i != int(os.environ['SLURM_ARRAY_TASK_ID']):
                    config_files.append([])
                    continue
                else:
                    cfg['first_run'] = True
                    cfg['resume_from'] = args.resume_from
            if 'LSB_JOBINDEX' in os.environ and int(os.environ['LSB_JOBINDEX']) >= 1:
                # LSF only allows one-based task IDs.
                if (i + 1) != int(os.environ['LSB_JOBINDEX']):
                    config_files.append([])
                    continue
                else:
                    cfg['first_run'] = True
                    cfg['resume_from'] = args.resume_from
            if args.local_rank == 0:
                cfg['name'] = f'{datetime.now().strftime("%y%m%d_%H%M")}_' \
                            f'{cfg["name"]}_{str(uuid.uuid4())[:5]}'
                cfg['work_dir'] = os.path.join('work_dirs', exp_name, cfg['name'])
                cfg['git_rev'] = get_git_hash()
                cfg['_base_'] = ['../../' + e for e in cfg['_base_']]
                cfg_out_file = f"{GEN_CONFIG_DIR}/{exp_name}/{cfg['name']}.json"
                os.makedirs(os.path.dirname(cfg_out_file), exist_ok=True)
                assert not os.path.isfile(cfg_out_file)
                with open(cfg_out_file, 'w') as of:
                    json.dump(cfg, of, indent=4)
                name_file = f"{os.path.join('work_dirs', exp_name)}/name_{cfg['name'][:-6]}_{str(i)}.txt"
                mmcv.mkdir_or_exist(os.path.abspath(cfg['work_dir']))
                with open(name_file, 'w') as of:
                    of.write(cfg['name'])
                    print(name_file)
                    print(cfg['name'])
            else:
                cfg_name_constant = f'{datetime.now().strftime("%y%m%d_%H%M")}_{cfg["name"]}'
                name_file = f"{os.path.join('work_dirs', exp_name)}/name_{cfg_name_constant}_{str(i)}.txt"
                print(name_file)
                while not os.path.exists(name_file):
                    time.sleep(5)
                with open(name_file, 'r') as f:
                    cfg['name'] = f.read()
                print('Recovered cfg[name]: ', cfg['name'])
                cfg['work_dir'] = os.path.join('work_dirs', exp_name, cfg['name'])
                cfg['git_rev'] = get_git_hash()
                cfg['_base_'] = ['../../' + e for e in cfg['_base_']]
                cfg_out_file = f"{GEN_CONFIG_DIR}/{exp_name}/{cfg['name']}.json"
            config_files.append(cfg_out_file)

    if args.machine == 'local':
        for i, cfg in enumerate(cfgs):
            if args.startup_test and cfg['seed'] != 0:
                continue
            if 'SLURM_ARRAY_TASK_ID' not in os.environ and ('LSB_JOBINDEX' not in os.environ or int(os.environ['LSB_JOBINDEX']) == 0) and i < args.seed_to_resume_from:
                continue
            # In case of job array, only run the configuration with number that corresponds to the task ID.
            if 'SLURM_ARRAY_TASK_ID' in os.environ:
                if i != int(os.environ['SLURM_ARRAY_TASK_ID']):
                    continue
            if 'LSB_JOBINDEX' in os.environ and int(os.environ['LSB_JOBINDEX']) >= 1:
                # LSF only allows one-based task IDs.
                if (i + 1) != int(os.environ['LSB_JOBINDEX']):
                    continue
            print('Run job {}'.format(cfg['name']))
            train.main([config_files[i]])
            torch.cuda.empty_cache()
    else:
        raise NotImplementedError(args.machine)
