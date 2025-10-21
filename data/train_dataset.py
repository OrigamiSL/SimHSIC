import os
import warnings
import copy
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import Pure_JM_distance
warnings.filterwarnings('ignore')


def linear_mix(x1, x2, a):
    return a * x1 + (1 - a) * x2


def binary_mix(x1, x2, a):
    m = np.random.choice([0, 1], size=x1.shape, p=[1 - a, a]).astype(np.float32)
    return m * x1 + (1 - m) * x2


def geometric_mix(x1, x2, a):
    return (x1 ** a) + (x2 ** (1 - a))


def cut_mix(x1, x2, a):
    H, W = x1.shape
    a_width = int(a * W)
    a_height = int(a * H)

    b_height = np.random.randint(0, H - a_height + 1)
    b_width = np.random.randint(0, W - a_width + 1)

    M = np.zeros((H, W))
    M[b_height:b_height + a_height, b_width:b_width + a_width] = 1
    x = M * x1 + (1 - M) * x2

    return x.astype(np.float32)


class Dataset_train(Dataset):
    def __init__(self, train_data, num_train, train_label, patch,
                 small_patch, sample_num, batch_size, aug_num):
        self.train_data = train_data
        self.num_train = num_train
        self.train_label = train_label
        self.batch_size = batch_size
        self.group_size = self.batch_size // len(self.num_train)
        self.random_list = []
        self.aug_num = aug_num
        self.patch = patch
        self.small_patch = small_patch
        self.sample_num = sample_num

        self.class_boundary = np.cumsum(self.num_train)
        for i in range(len(self.num_train)):
            total_instance = np.linspace(0, self.num_train[i] - 1, self.num_train[i], dtype=int)  # 生成采样空间
            final_choice = 0
            final_JM_score = 0
            for j in range(50):
                current_choice = np.random.choice(total_instance, sample_num, replace=False)
                current_list = []
                for k in range(self.sample_num):
                    if i == 0:
                        instance = self.train_data[current_choice[k]]  # N, P * P
                        N, D = instance.shape
                        current_list.append(instance[:, D // 2])
                    else:
                        instance = self.train_data[current_choice[k] +
                                                   self.class_boundary[i - 1]]
                        N, D = instance.shape
                        current_list.append(instance[:, D // 2])
                current_group = np.stack(current_list, axis=0)
                JM_score = Pure_JM_distance(current_group)
                if JM_score > final_JM_score:
                    final_choice = current_choice
                    final_JM_score = JM_score
            self.random_list.append(final_choice)  # 采样
        self.__read_data__()

    def __read_data__(self):
        train_data = []
        train_label = []
        support_data = []
        support_label = []
        for i in range(len(self.num_train)):
            for j in range(self.sample_num):
                if i == 0:
                    instance = torch.tensor(self.train_data[self.random_list[i][j]]).unsqueeze(0)  # 1, N, P * P
                    with (torch.no_grad()):
                        instance = instance.contiguous().view(1, instance.shape[1], self.patch,
                                                              self.patch)  # 1, N, P, P
                        t_unfold = torch.nn.Unfold(kernel_size=(self.small_patch, self.small_patch), stride=(1, 1))

                        instance = t_unfold(instance).contiguous().view(
                            1, instance.shape[1], self.small_patch ** 2, -1).squeeze(0)  # N, p * p, L
                        N, P, L = instance.shape
                        center_instance = instance[:, :, L // 2]
                        instance = instance.detach().cpu().numpy()
                        center_instance = center_instance.detach().cpu().numpy()
                        support_data.append(center_instance)
                        support_label.append(self.train_label[self.random_list[i][j]])
                    for k in range(instance.shape[-1]):
                        train_data.append(instance[:, :, k])
                        train_label.append(self.train_label[self.random_list[i][j]])
                else:
                    instance = (torch.tensor(self.train_data[self.class_boundary[i - 1] + self.random_list[i][j]]).
                                unsqueeze(0))  # 1, N, P * P
                    with (torch.no_grad()):
                        instance = instance.contiguous().view(1, instance.shape[1], self.patch, self.patch)
                        t_unfold = torch.nn.Unfold(kernel_size=(self.small_patch, self.small_patch), stride=(1, 1))
                        instance = t_unfold(instance).contiguous().view(
                            1, instance.shape[1], self.small_patch ** 2, -1).squeeze(0)  # N, p * p, L
                        N, P, L = instance.shape
                        center_instance = instance[:, :, L // 2]
                        instance = instance.detach().cpu().numpy()
                        center_instance = center_instance.detach().cpu().numpy()
                        support_data.append(center_instance)
                        support_label.append(self.train_label[self.class_boundary[i - 1] + self.random_list[i][j]])
                    for k in range(instance.shape[-1]):
                        train_data.append(instance[:, :, k])
                        train_label.append(self.train_label[self.class_boundary[i - 1] + self.random_list[i][j]])
        self.f_train_data = np.stack(train_data, axis=0)  # B个[N, p * p]->[B, N, p * p]
        self.f_train_label = np.stack(train_label, axis=0)
        self.support_data = np.stack(support_data, axis=0)
        self.support_label = np.stack(support_label, axis=0)

        c_num = self.f_train_data.shape[0] // len(self.num_train)  # 采样以后的训练集每个类有几个例子
        self.class_i_list = np.linspace(0, c_num - 1, c_num, dtype=int)
        self.train_class_boundary = np.ones([len(self.num_train)], dtype=int) * c_num
        self.train_class_boundary = np.cumsum(self.train_class_boundary)

    def __getitem__(self, index):
        instance_data = []
        instance_label = []
        mix_index = np.linspace(0, 3, 4, dtype=int)
        for i in range(len(self.num_train)):
            random_index = np.random.choice(self.class_i_list, self.group_size, replace=False)
            for j in range(self.group_size):
                if i == 0:
                    instance_data.append(self.f_train_data[random_index[j]])
                    instance_label.append(self.f_train_label[random_index[j]])
                    for k in range(self.aug_num):  # Mixup
                        sample_index = np.random.choice(self.class_i_list, 2, replace=False)
                        rand_num = np.random.rand()  # 随机数alpha (0~1)
                        sample1 = self.f_train_data[sample_index[0]]  # x2
                        sample2 = self.f_train_data[sample_index[1]]  # x2
                        mix_method = np.random.choice(mix_index, 1, replace=False)
                        sample_mix = 0
                        if mix_method[0] == 0:
                            sample_mix = linear_mix(sample1, sample2, rand_num)  # alpha * x1 + (1 - alpha) * x2
                        elif mix_method[0] == 1:
                            sample_mix = binary_mix(sample1, sample2, rand_num)
                        elif mix_method[0] == 2:
                            sample_mix = geometric_mix(sample1, sample2, rand_num)
                        elif mix_method[0] == 3:
                            sample_mix = cut_mix(sample1, sample2, rand_num)
                        # zero augmentation
                        rand_zero = np.random.rand(sample_mix.shape[1]).reshape(-1)
                        rand_zero = np.expand_dims(rand_zero, axis=0)
                        rand_zero = np.repeat(rand_zero, sample_mix.shape[0], axis=0)
                        zero_matrix = np.zeros_like(sample_mix)
                        sample_mix = np.where(rand_zero < 0.8, sample_mix, zero_matrix)
                        instance_data.append(sample_mix)
                        instance_label.append(self.f_train_label[random_index[j]])
                else:
                    instance_data.append(self.f_train_data[self.train_class_boundary[i - 1] + random_index[j]])
                    instance_label.append(self.f_train_label[self.train_class_boundary[i - 1] + random_index[j]])
                    for k in range(self.aug_num):
                        sample_index = np.random.choice(self.class_i_list, 2, replace=False)
                        rand_num = np.random.rand()
                        sample1 = self.f_train_data[self.train_class_boundary[i - 1] + sample_index[0]]
                        sample2 = self.f_train_data[self.train_class_boundary[i - 1] + sample_index[1]]
                        mix_method = np.random.choice(mix_index, 1, replace=False)
                        sample_mix = 0
                        if mix_method[0] == 0:
                            sample_mix = linear_mix(sample1, sample2, rand_num)  # alpha * x1 + (1 - alpha) * x2
                        elif mix_method[0] == 1:
                            sample_mix = binary_mix(sample1, sample2, rand_num)
                        elif mix_method[0] == 2:
                            sample_mix = geometric_mix(sample1, sample2, rand_num)
                        elif mix_method[0] == 3:
                            sample_mix = cut_mix(sample1, sample2, rand_num)
                        # zero augmentation
                        rand_zero = np.random.rand(sample_mix.shape[1]).reshape(-1)
                        rand_zero = np.expand_dims(rand_zero, axis=0)
                        rand_zero = np.repeat(rand_zero, sample_mix.shape[0], axis=0)
                        zero_matrix = np.zeros_like(sample_mix)
                        sample_mix = np.where(rand_zero < 0.8, sample_mix, zero_matrix)
                        instance_data.append(sample_mix)
                        instance_label.append(self.f_train_label[self.train_class_boundary[i - 1] + random_index[j]])
        instance_data = np.stack(instance_data, axis=0)
        instance_label = np.stack(instance_label, axis=0)
        return instance_data, instance_label

    def __len__(self):
        return len(self.train_data) // self.batch_size

    def get_support_set(self):
        return self.support_data, self.support_label

    def get_non_train_index(self):
        total_index = []
        for i in range(len(self.num_train)):
            for j in range(self.sample_num):
                if i == 0:
                    total_index.append(self.random_list[i][j])
                else:
                    total_index.append(self.class_boundary[i - 1] + self.random_list[i][j])
        return np.array(total_index)
