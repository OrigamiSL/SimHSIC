import torch
import numpy as np
import math
from loguru import logger


def Contrastive_Loss(x, num_classes, temp):
    B, D = x.shape
    G = B // num_classes
    x = x / torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True)).expand(B, D)
    similarity_matrix = torch.einsum("bd,sd->bs", x, x)  # B, B
    similarity_matrix = torch.exp(similarity_matrix / temp)
    # CC CM CO
    # MC MM MO
    # OC OM OO
    loss = 0
    mask = torch.ones_like(similarity_matrix) - torch.eye(B, B).to(similarity_matrix.device)
    similarity_matrix = similarity_matrix * mask
    for i in range(num_classes):
        for j in range(G):
            pos = torch.sum(similarity_matrix[i * G + j, i * G: (i + 1) * G])
            pos_neg = torch.sum(similarity_matrix[i * G + j])
            loss += -torch.log(pos / pos_neg)
    return loss / B


def Contrastive_Score(x_repr, support_repr, num_classes, temp):
    B, D = x_repr.shape
    S, _ = support_repr.shape
    x_repr = x_repr / torch.sqrt(torch.sum(x_repr ** 2, dim=-1, keepdim=True)).expand(B, D)
    support_repr = support_repr / torch.sqrt(torch.sum(support_repr ** 2, dim=-1, keepdim=True)).expand(S, D)
    similarity_matrix = torch.einsum("bd,sd->bs", x_repr, support_repr) / temp  # B, B
    similarity_matrix = similarity_matrix.contiguous().view(B, num_classes, S // num_classes)
    score = torch.softmax(torch.sum(similarity_matrix, dim=-1), dim=-1)
    return score


# -------------------------------------------------------------------------------
def chooose_train_and_test_point(train_data, test_data, true_data, num_classes):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}
    # -------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data == (i + 1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class

    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]]  # (695,2)
    total_pos_train = total_pos_train.astype(int)
    # --------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data == (i + 1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]]  # (9671,2)
    total_pos_test = total_pos_test.astype(int)
    total_pos_true = None
    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true


# -------------------------------------------------------------------------------
def mirror_hsi(height, width, band, input_normalize, patch=5):
    padding = patch // 2
    mirror_hsi = np.zeros((height + 2 * padding, width + 2 * padding, band), dtype=float)
    mirror_hsi[padding:(padding + height), padding:(padding + width), :] = input_normalize
    for i in range(padding):
        mirror_hsi[padding:(height + padding), i, :] = input_normalize[:, padding - i - 1, :]
    for i in range(padding):
        mirror_hsi[padding:(height + padding), width + padding + i, :] = input_normalize[:, width - 1 - i, :]
    for i in range(padding):
        mirror_hsi[i, :, :] = mirror_hsi[padding * 2 - i - 1, :, :]
    for i in range(padding):
        mirror_hsi[height + padding + i, :, :] = mirror_hsi[height + padding - 1 - i, :, :]

    logger.info("**************************************************")
    logger.info("patch is : {}".format(patch))
    logger.info(
        "mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0], mirror_hsi.shape[1], mirror_hsi.shape[2]))
    logger.info("**************************************************")
    return mirror_hsi


# -------------------------------------------------------------------------------
def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i, 0]
    y = point[i, 1]
    temp_image = mirror_image[x:(x + patch), y:(y + patch), :]
    return temp_image


# -------------------------------------------------------------------------------
def train_and_test_data(mirror_image, band, train_point, test_point, true_point, patch=9, s_patch=3):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_train_of_test = np.zeros((train_point.shape[0], s_patch, s_patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], s_patch, s_patch, band), dtype=float)
    for i in range(train_point.shape[0]):
        x_train[i, :, :, :] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
        x_train_of_test[i, :, :, :] = gain_neighborhood_pixel(mirror_image, train_point, i, s_patch)
    for j in range(test_point.shape[0]):
        x_test[j, :, :, :] = gain_neighborhood_pixel(mirror_image, test_point, j, s_patch)
    x_test = np.concatenate([x_train_of_test, x_test], axis=0)
    logger.info("x_train shape = {}, type = {}".format(x_train.shape, x_train.dtype))
    logger.info("x_test  shape = {}, type = {}".format(x_test.shape, x_test.dtype))
    logger.info("**************************************************")

    return (x_train.reshape(train_point.shape[0], patch * patch, band),
            x_test.reshape(test_point.shape[0] + train_point.shape[0], s_patch * s_patch, band))


# -------------------------------------------------------------------------------
# y_train, y_test
def train_and_test_label(number_train, number_test, number_true, num_classes):
    y_train = []
    y_test = []
    # y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    y_test = np.concatenate([y_train, y_test], axis=0)
    logger.info("y_train: shape = {} ,type = {}".format(y_train.shape, y_train.dtype))
    logger.info("y_test: shape = {} ,type = {}".format(y_test.shape, y_test.dtype))
    logger.info("**************************************************")
    return y_train, y_test


# -------------------------------------------------------------------------------
class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


# -------------------------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, pred.squeeze()


# -------------------------------------------------------------------------------
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float64)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / (np.sum(matrix[i, :]) + 1e-6)
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


def JM_distance(x_instance, support_instance, num_classes):
    B, D = x_instance.shape
    S, _ = support_instance.shape
    x_mean = torch.mean(x_instance, dim=-1)
    x_std = torch.std(x_instance, dim=-1)
    support_mean = torch.mean(support_instance, dim=-1)
    support_std = torch.std(support_instance, dim=-1)
    JM_Matrix = torch.zeros([B, S]).to(x_instance.device)
    for i in range(B):
        for j in range(S):
            JM_Matrix[i, j] = ((1 / 8) * (x_mean[i] - support_mean[j]) ** 2 * (2 / (x_std[i] + support_std[j])) +
                               (1 / 2) * torch.log(
                        (x_std[i] + support_std[j]) / 2 / torch.sqrt(x_std[i] * support_std[j])))
            JM_Matrix[i, j] = torch.sqrt(2 * (1 - torch.exp(-JM_Matrix[i, j])))
    JM_Matrix = JM_Matrix.contiguous().view(B, num_classes, S // num_classes)
    score = torch.softmax(1 / torch.mean(JM_Matrix, dim=-1), dim=-1)
    return score


def Pure_JM_distance(x_instance):
    B, D = x_instance.shape
    x_mean = np.mean(x_instance, axis=-1)
    x_std = np.std(x_instance, axis=-1)
    JM_Matrix = np.zeros([B, B])
    for i in range(B):
        for j in range(B):
            JM_Matrix[i, j] = ((1 / 8) * (x_mean[i] - x_mean[j]) ** 2 * (2 / (x_std[i] + x_std[j])) +
                               (1 / 2) * np.log((x_std[i] + x_std[j]) / 2 / np.sqrt(x_std[i] * x_std[j])))
            JM_Matrix[i, j] = np.sqrt(2 * (1 - np.exp(-JM_Matrix[i, j])))
    score = np.mean(JM_Matrix)
    return score
