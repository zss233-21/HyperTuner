import asyncio
import json
import os
import sys
import time
import yaml
import re
import numpy as np
from datetime import datetime
from pathlib import Path
from lib.other import parse_cmd, get_default
from lib.optimizer import create_optimizer


def run_playbook(playbook_path, tags='all', **extra_vars):
    vars_json_str = json.dumps(extra_vars)
    command = f'ansible-playbook {playbook_path} --extra-vars=\'{vars_json_str}\' --tags={tags}'
    print(command)
    os.system(command)


def translate_config_to_numeric(sample_config, app_setting):
    config = dict(app_setting)
    # default configs, need to transform category into values
    sample_config_v = {}
    for k, v in sample_config.items():
        v_range = config[k].get('range')
        if v_range:
            sample_config_v[k] = v_range.index(v)
        else:
            sample_config_v[k] = v
    return sample_config_v


def find_exist_task_result():
    task_id = -1
    regexp = re.compile(r'(\d+)_.+')
    if result_dir.exists():
        for p in result_dir.iterdir():
            if p.is_file():
                res = regexp.match(p.name)
                if res:
                    task_id = max(task_id, int(res.group(1)))
    return None if task_id == -1 else task_id


def divide_config(sampled_config, os_setting, app_setting):
    for k in sampled_config.keys():
        if type(sampled_config[k]) is bool:
            # make sure no uppercase 'True/False' literal in result
            sampled_config[k] = str(sampled_config[k]).lower()
        elif type(sampled_config[k]) is np.float64:
            sampled_config[k] = float(sampled_config[k])

    sampled_os_config = dict(
        ((k, v) for k, v in sampled_config.items() if k in os_setting)
    )
    sampled_config = dict(
        ((k, v) for k, v in sampled_config.items() if k in app_setting)
    )
    return sampled_os_config, sampled_config


def _print(msg):
    print(f'[{datetime.now()}] {test_config.task_name} - {msg}')
    # print('[' + datetime.now() + ']')


def dataenergy(path1, path2, path3):
    data = []
    energy = 0
    paths = [path2, path3, path1]
    for path in paths:
        filename = path
        lnum = 0
        pkg_J, cor_J, ram_J = [], [], []
        with open(filename, 'r') as f1:
            # 将txt中的数据逐行存到列表lines里 lines的每一个元素对应于txt中的一行。
            # 然后将每个元素中的不同信息提取出来
            lines = f1.readlines()
            # i变量，由于这个txt存储时有空行，所以增只读偶数行，主要看txt文件的格式，一般不需要
            # j用于判断读了多少条，step为画图的X轴
            i = 0
            for line in lines:
                lnum += 1
                if lnum > 1:
                    t = line.split('\t')
                    while len(t) == 3:
                        l0 = len(t[0])
                        if 7 > l0 > 3:
                            l2 = len(t[2])
                            if 0 < l2 < 7:
                                pkg_J.append(float(t[0]))
                                cor_J.append(float(t[1]))
                                ram_J.append(float(t[2]))
                                i = i + 1
            energy = energy + sum(pkg_J) + sum(cor_J) + sum(ram_J)
            runtime = len(pkg_J)
    data = [runtime, energy]
    return data


def objective():
    # data = []
    start_time = time.time()
    common = 'python3.8 /home/zss/PycharmProjects/pythonProject/run-playbook-without-kernel.py'
    os.system(common)
    end_time = time.time()
    exe_time = end_time - start_time
    data = dataenergy('/home/zss/PycharmProjects/pythonProject/result/energy-test-155.txt',
                      '/home/zss/PycharmProjects/pythonProject/result/energy-test-153.txt',
                      '/home/zss/PycharmProjects/pythonProject/result/energy-test-152.txt')
    print("the execution time is", exe_time)
    runtime = -1.0 * data[0]
    power = -1.0 * data[1]
    obj = [runtime, power]
    # f1 = open("/home/zss/PycharmProjects/pythonProject/data/lenet-no-kernel(7-16).csv", "a")
    # print(-runtime, -power, file=f1)
    return obj


async def main(test_config, init_id,os_setting, app_setting):  # os_setting,
    global proj_root

    # BO初始化 参数空间
    optimizer = create_optimizer(
        test_config.optimizer.name,  # mbo-ei name , init mbo-without-kernel
        {
            **app_setting
        },
        extra_vars=test_config.optimizer.extra_vars
    )
    if hasattr(optimizer, 'set_status_file'):
        optimizer.set_status_file(result_dir / 'optimizer_status')
    task_id = init_id
    while task_id < test_config.optimizer.iter_limit:
        task_id += 1
        # - sample config
        if task_id == 0:  # use default config
            sampled_config_numeric, sampled_config = None, get_default(app_setting, os_setting)
            # sampled_config_numeric, sampled_config = None, get_default(app_setting, None)
            f = open("/home/zss/PycharmProjects/pythonProject/param_list.txt", "w")
            print(sampled_config['cpu_freq'], sampled_config['dropout_rate'], sampled_config['learn_rate'],
                  sampled_config['batch_size'], task_id, sampled_config['inter_op'], sampled_config['intra_op'], file=f)
        else:
            try:
                sampled_config_numeric, sampled_config = optimizer.get_conf()
            except StopIteration:
                # all configuration emitted
                return
        f = open("/home/zss/PycharmProjects/pythonProject/param_list.txt", "w")
        print(sampled_config['cpu_freq'], sampled_config['dropout_rate'], sampled_config['learn_rate'],
              sampled_config['batch_size'], task_id, sampled_config['inter_op'], sampled_config['intra_op'],
              file=f)  # sampled_config['top_words'], sampled_config['max_review_length']
        # sampled_config['inter_op'], sampled_config['intra_op']

        # 格式转换
        # sampled_os_config, sampled_app_config = divide_config(
        #     sampled_config,
        #     os_setting=os_setting,
        #     app_setting=app_setting
        # )
        # if tune_app is off, just give sample_app_config a default value

        # 生成配置
        # os_config_path = result_dir / f'{task_id}_os_config.yml'
        # os_config_path.write_text(
        #     yaml.dump(sampled_os_config, default_flow_style=False)
        # )
        app_config_path = result_dir / f'{task_id}_app_config.yml'
        app_config_path.write_text(
            yaml.dump(sampled_config, default_flow_style=False)
        )
        _print(f'{task_id}: app_config generated.')

        # 配置参数传给 set_os.yml  、CNN
        # 启动run_test.py 获取指标
        # 每一轮指标加到time（power）_list中
        # metric[time_list,power_list]
        # f = open("/home/zss/PycharmProjects/pythonProject/lenet/test.csv", "a")
        # print(task_id, file=f)
        metric = objective()
        # metric = [350, 5368.270000000002]
        metric_result = []

        # after
        if task_id != 0:  # not adding default info, 'cause default cannot convert to numeric form
            metric_result.append(metric[0])
            metric_result.append(metric[1])
            optimizer.add_observation(
                (sampled_config_numeric, metric_result)
                #      (sampled_config_numeric, metric[1])
            )
            # f3 = open("/home/zss/PycharmProjects/pythonProject/data/output_rnn.csv", "a") print(sampled_config[
            # 'cpu_freq'], sampled_config['dropout_rate'], sampled_config['learn_rate'], sampled_config[
            # 'batch_size'], sampled_config['inter_op'], sampled_config['intra_op'], sampled_config[
            # 'dirty_background_ratio'], sampled_config['dirty_expire_centisecs'], sampled_config['dirty_ratio'],
            # sampled_config['max_map_count'], sampled_config['netdev_max_backlog'], sampled_config['nr_requests'],
            # sampled_config['read_ahead_kb'], sampled_config['rq_affinity'], sampled_config['somaxconn'],
            # sampled_config['swappiness'], sampled_config['tcp_abort_on_overflow'], sampled_config[
            # 'tcp_max_syn_backlog'], sampled_config['vfs_cache_pressure'], sampled_config[
            # 'tcp_slow_start_after_idle'], -1 * metric[0], -1 * metric[1], file=f3)
            # f2 = open("/home/zss/PycharmProjects/pythonProject/out-vgg-no-kernel.txt", "a")
            # print(sampled_config['cpu_freq'], sampled_config['dropout_rate'], sampled_config['learn_rate'],
            #       sampled_config['batch_size'], task_id, sampled_config['inter_op'], sampled_config['intra_op'],
            #       -1 * metric[0], -1 * metric[1], file=f2)
            if hasattr(optimizer, 'dump_state'):
                optimizer.dump_state(result_dir / f'{task_id}_optimizer_state')


# -------------------------------------------------------------------------------------------------------
test_config = parse_cmd()
assert test_config is not None

# calculate paths
#  Hdconfigor
proj_root = Path(__file__, '../../..').resolve()
#  Hdconfigor/target/hbase
db_dir = proj_root / f'target/{test_config.target}'
result_dir = db_dir / f'results/{test_config.task_name}'

# set_osplaybook_path = db_dir / 'playbook/set_os.yml'
# set_kernelplaybook_path = db_dir / 'playbook/set_kernel.yml'

app_setting_path = proj_root / f'target/{test_config.target}/app_configs_info.yml'
os_setting_path = proj_root / f'target/{test_config.target}/os_configs_info.yml'

init_id = -1

# check existing results, find minimum available task_id
exist_task_id = find_exist_task_result()

if exist_task_id is not None:
    _print(f'previous results found, with max task_id={exist_task_id}')
    policy = test_config.exist
    if policy == 'delete':
        for file in sorted(result_dir.glob('*')):
            file.unlink()
        _print('all deleted')
    elif policy == 'continue':
        _print(f'continue with task_id={exist_task_id + 1}')
        init_id = exist_task_id
    else:
        _print('set \'exist\' to \'delete\' or \'continue\' to specify what to do, exiting...')
        sys.exit(0)

# create dirs
result_dir.mkdir(parents=True, exist_ok=True)

# dump test configs
(result_dir / 'test_config.yml').write_text(
    yaml.dump(test_config, default_flow_style=False)
)
_print('test_config.yml dumped')

# read parameters for tuning
app_setting = yaml.load(app_setting_path.read_text())  # pylint: disable=E1101
os_setting = yaml.load(os_setting_path.read_text())  # pylint: disable=E1101

# event loop, main() is async
loop = asyncio.get_event_loop()
loop.run_until_complete(
    main(
        test_config=test_config,
        init_id=init_id,
        app_setting=app_setting,
        os_setting=os_setting
    )
)
loop.close()
