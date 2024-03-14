import os
import copy
import json
import time
import shutil
from d4rl import infos

import torch

import numpy as np


ENV_OBS_RANGE = {
    'walker2d-v0': dict(
        max=[1.8164345026016235, 0.999911367893219, 0.5447346568107605, 0.7205190062522888,
             1.5128496885299683, 0.49508699774742126, 0.6822911500930786, 1.4933640956878662,
             9.373093605041504, 5.691765308380127, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        min=[0.800006091594696, -0.9999997019767761, -3.006617546081543, -2.9548180103302,
             -1.72023344039917, -2.9515464305877686, -3.0064914226531982, -1.7654582262039185,
             -6.7458906173706055, -8.700752258300781, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0,
             -10.0]
    ),
    'hopper-v0': dict(
        max=[1.7113906145095825, 0.1999576985836029, 0.046206455677747726, 0.10726844519376755,
             0.9587112665176392, 5.919354438781738, 3.04956316947937, 6.732881546020508,
             7.7671966552734375, 10.0, 10.0],
        min=[0.7000009417533875, -0.1999843567609787, -1.540910243988037, -1.1928397417068481,
             -0.9543644189834595, -1.6949318647384644, -5.237359523773193, -6.2852582931518555,
             -10.0, -10.0, -10.0]
    ),
    'halfcheetah-v0': dict(
        max=[1.600443959236145,22.812137603759766, 1.151809811592102, 0.949776291847229,
             0.9498141407966614, 0.8997246026992798, 1.1168793439865112, 0.7931482791900635,
             16.50477409362793, 5.933143138885498, 13.600515365600586, 27.84033203125,
             30.474760055541992, 30.78533935546875, 30.62249755859375, 37.273799896240234,
             31.570491790771484],
        min=[-0.6028550267219543, -3.561767339706421, -0.7100794315338135, -1.0610754489898682,
             -0.6364201903343201, -1.2164583206176758, -1.2236766815185547, -0.7376371026039124,
             -3.824833869934082, -5.614060878753662, -12.930273056030273, -29.38336944580078,
             -31.534399032592773, -27.823902130126953, -32.996246337890625, -30.887380599975586,
             -30.41145896911621]
    ),
    'walker2d-v2': dict(
        max=[1.731740117073059, 0.9998167157173157, 0.40092915296554565, 0.5324582457542419,
             1.4848185777664185, 0.48324301838874817, 0.5194370150566101, 1.515627145767212,
             7.567550182342529, 3.599123239517212, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, ],
        min=[0.8000273108482361, -0.9999956488609314, -2.173219919204712, -2.786832571029663,
             -1.6133389472961426, -2.5313565731048584, -2.793543577194214, -1.6024911403656006,
             -5.634458065032959, -6.61629581451416, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0]
    ),
    'hopper-v2': dict(
        max=[1.8055554628372192, 0.19994886219501495, 0.05772435665130615, 0.12416191399097443,
             0.9748091697692871, 5.633016586303711, 3.3520665168762207, 7.937284469604492, 10.0,
             10.0, 10.0],
        min=[0.7000075578689575, -0.19999995827674866, -1.7813633680343628, -2.1054952144622803,
             -0.9721308946609497, -2.2943766117095947, -5.959066867828369, -8.406386375427246,
             -10.0, -10.0, -10.0]
    ),
    'halfcheetah-v2': dict(
        max=[1.19423508644104, 16.12965202331543, 1.1145673990249634, 0.8977003693580627,
             0.9202039241790771, 0.8960233926773071, 1.0794918537139893, 0.7445417046546936,
             14.500947952270508, 4.665963172912598, 11.089605331420898, 24.246370315551758,
             31.478853225708008, 23.721609115600586, 30.441913604736328, 33.67640686035156,
             26.321840286254883],
        min=[-0.5985506772994995, -3.512727737426758, -0.6979433298110962, -0.9877811074256897,
             -0.6536482572555542, -1.2188079357147217, -1.3276208639144897, -0.7179604768753052,
             -3.8287100791931152, -4.854346752166748, -9.264730453491211, -28.212547302246094,
             -32.98865509033203, -30.40522575378418, -30.605560302734375, -29.96497917175293,
             -28.141843795776367]
    ),
}

ENV_NAME_MAPPING = {
    'walker2d-random-v0': 'w2d-r',
    'walker2d-medium-v0': 'w2d-m',
    'walker2d-medium-replay-v0': 'w2d-m-re',
    'walker2d-medium-expert-v0': 'w2d-m-e',
    'walker2d-expert-v0': 'w2d-e',
    'hopper-random-v0': 'hop-r',
    'hopper-medium-v0': 'hop-m',
    'hopper-medium-replay-v0': 'hop-m-re',
    'hopper-medium-expert-v0': 'hop-m-e',
    'hopper-expert-v0': 'hop-e',
    'halfcheetah-random-v0': 'che-r',
    'halfcheetah-medium-v0': 'che-m',
    'halfcheetah-medium-replay-v0': 'che-m-re',
    'halfcheetah-medium-expert-v0': 'che-m-e',
    'halfcheetah-expert-v0': 'che-e',

    'walker2d-random-v2': 'w2d2-r',
    'walker2d-medium-v2': 'w2d2-m',
    'walker2d-medium-replay-v2': 'w2d2-m-re',
    'walker2d-medium-expert-v2': 'w2d2-m-e',
    'walker2d-expert-v2': 'w2d2-e',
    'hopper-random-v2': 'hop2-r',
    'hopper-medium-v2': 'hop2-m',
    'hopper-medium-replay-v2': 'hop2-m-re',
    'hopper-medium-expert-v2': 'hop2-m-e',
    'hopper-expert-v2': 'hop2-e',
    'halfcheetah-random-v2': 'che2-r',
    'halfcheetah-medium-v2': 'che2-m',
    'halfcheetah-medium-replay-v2': 'che2-m-re',
    'halfcheetah-medium-expert-v2': 'che2-m-e',
    'halfcheetah-expert-v2': 'che2-e',
    'Ant-v4': 'ant',
    'InvertedPendulum-v4': 'pendulum',
    'InvertedDoublePendulum-v4': 'doublependulum',
    'Reacher-v4': 'reacher',
    'Swimmer-v4': 'swimmer',

    'unknown': 'unknown'
}



def standardization(x, mean, std, eps=1e-3):
    return (x - mean) / (std + eps)

def reverse_standardization(x, mean, std, eps=1e-3):
    return ((std + eps) * x) + mean

def normalize(x, min, max):
    x = (x - min)/(max - min)
    return x


def denormalize(x, min, max):
    x = x * (max - min) + min
    return x

def tensor(x, device='cpu'):
    if isinstance(x, torch.Tensor):
        return x
    dtype = torch.uint8 if x.dtype == np.uint8 else torch.float32
    tensor = torch.tensor(data=x, dtype=dtype, device=device)
    return tensor.float()


def clamp(x, vec_min, vec_max):
    if isinstance(vec_min, list):
        vec_min = torch.Tensor(vec_min).to(x.device)
    if isinstance(vec_max, list):
        vec_max = torch.Tensor(vec_max).to(x.device)

    assert isinstance(vec_min, torch.Tensor) and isinstance(vec_max, torch.Tensor)
    x = torch.max(x, vec_min)
    x = torch.min(x, vec_max)
    return x


def copy_file(src, des):
    try:
        shutil.copy(src, des)
        print("File copied successfully.")

    # If source and destination are same
    except shutil.SameFileError:
        print("Source and destination represents the same file.")

    # If there is any permission issue
    except PermissionError:
        print("Permission denied.")

    # For other errors
    except:
        print("Error occurred while copying file.")


def get_stats_from_ckpt(checkpoint):
    stats_filename = copy.copy(checkpoint)
    stats_filename = stats_filename.replace('model_', 'stats_')
    stats_filename = stats_filename.replace('.pt', '.npz')
    if os.path.isfile(stats_filename):
        data = np.load(stats_filename)
        mean, std = data['mean'], data['std']
    else:
        print('[WARNING] Not found statistical of replay buffer.')
        mean, std = None, None

    return mean, std


def make_checkpoint_list(ckpt_path, n_seeds_want_to_test, ckpt_steps):
    print("[INFO] Finding checkpoints...")
    if os.path.isfile(ckpt_path):
        assert n_seeds_want_to_test == 1
        ckpt_list = [ckpt_path]
        print("\tEvaluating with single checkpoint.")
    elif os.path.isdir(ckpt_path):
        entries = os.listdir(ckpt_path)
        entries.sort()
        print("\tFound %d experiments." % (len(entries)))
        ckpt_list = []
        for entry in entries:
            ckpt_file = os.path.join(ckpt_path, entry, ckpt_steps)
            if not os.path.isfile(ckpt_file):
                print("\tCannot find checkpoint {} in {}".format(ckpt_steps, ckpt_file))
            else:
                ckpt_list.append(ckpt_file)

        print('\tFound {} checkpoints.'.format(len(ckpt_list)))
        if len(ckpt_list) < n_seeds_want_to_test:
            print("\tWARNING: Number of found checkpoints less than requirement")
    else:
        print("\tPath doesn't exist: ", ckpt_path)
        raise ValueError

    return ckpt_list


def count_num_seeds_in_path(ckpt_path, ckpt_steps):
    assert os.path.isdir(ckpt_path)
    entries = os.listdir(ckpt_path)
    entries.sort()
    ckpt_list = []
    for entry in entries:
        ckpt_file = os.path.join(ckpt_path, entry, ckpt_steps)
        if not os.path.isfile(ckpt_file):
            print("[WARNING] Cannot find checkpoint {} in {}".format(ckpt_steps, ckpt_file))
        else:
            ckpt_list.append(ckpt_file)

    return len(ckpt_list)


def set_name_wandb_project(dataset):
    project_name = None
    if 'hopper' in dataset:
        project_name = 'HOPPER-VQ'
    elif 'walker' in dataset:
        project_name = 'WALKER-VQ'
    elif 'halfcheetah' in dataset:
        project_name = 'CHEETAH-VQ'
    elif 'Ant-v4' == dataset:
        project_name = 'ANT-VQ'
    elif 'InvertedPendulum-v4' == dataset:
        project_name = 'PENDULUM-VQ'
    elif 'Reacher-v4' == dataset:
        project_name = 'REACHER-VQ'
    elif 'InvertedDoublePendulum-v4' == dataset:
        project_name = 'DOUBLEPENDULUM-VQ'
    elif 'Swimmer-v4' == dataset:
        project_name = 'SWIMMER-VQ'
    else:
        raise NotImplementedError

    return project_name


class EvalLogger():
    def __init__(self, ckpt, eval_logdir, prefix="eval", eval_args=None):

        # Extract required information
        assert os.path.isfile(ckpt)
        self.eval_args = eval_args


        # Construct new writer
        timestamp = time.localtime()
        timestamp = time.strftime("%m_%d-%H_%M_%S", timestamp)
        self.filename = prefix + '_' + '_' + timestamp + '.txt'
        self.logfile_no_date = os.path.join(eval_logdir, self.filename)
        if not os.path.isdir(os.path.join(eval_logdir, 'with_date')):
            os.makedirs(os.path.join(eval_logdir, 'with_date'))

        self.logfile = os.path.join(eval_logdir, 'with_date', self.filename)
        self.writer = open(self.logfile, "w")

        if eval_args is not None:
            self.write_eval_args()

    def write_eval_args(self):
        self.writer.write("\n\n********* EVALUATION PARAMS *********\n\n")
        for key, val in vars(self.eval_args).items():
            self.writer.write("\t\t%s: %s\n" % (key, val))

        self.writer.write("\n\n*************************************\n\n")


    def log(self, attack_type, attack_epsilon, attack_iteration, unorm_score, norm_score):
        if attack_type == 'clean':
            self.writer.write("Clean performance: unorm_score=%.3f, norm = %.2f\n" %
                              (unorm_score, norm_score)
                              )
        else:
            self.writer.write("Attack=%s - epsilon=%.4f, n_iters=%d: unorm_score=%.3f, norm = %.2f\n"
                              % (attack_type.upper(), attack_epsilon, attack_iteration,
                                 unorm_score, norm_score)
                              )
    def print(self, text):
        print(text)
        self.writer.write(text)


    def close(self):
        self.writer.close()



class LinearSchedule:
    def __init__(self, start_val, end_val=None, n_steps=None, start_step=1):
        if end_val is None:
            end_val = start_val
            n_steps = 1
        self.inc = (end_val - start_val) / float(n_steps)
        self.current = start_val
        self.end_val = end_val
        if end_val > start_val:
            self.bound = min
        else:
            self.bound = max
        self.total_steps = 0
        self.start_step = start_step

    def __call__(self, steps=1):
        val = self.current
        self.total_steps += 1
        if self.total_steps >= self.start_step:
            self.current = self.bound(self.current + self.inc * steps, self.end_val)
        return val



