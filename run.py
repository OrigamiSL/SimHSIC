import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from scipy.io import loadmat
from model.Network import Network
from sklearn.metrics import confusion_matrix
from data.train_dataset import Dataset_train
from utils import *
import numpy as np
import time
import os
from collections import Counter
from loguru import logger

logger.add(
    './info.log',
    format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
    level="INFO",
)

parser = argparse.ArgumentParser(description='Multiaspect Hyperspectral Image Classification Transformer')
parser.add_argument('--dataset', choices=['Indian', 'Pavia', 'Houston', 'Salinas'],
                    default='Indian', help='dataset to use')
parser.add_argument('--flag_test', choices=['test', 'train'], default='train', help='testing mark')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=2024, help='number of seed')
parser.add_argument('--batch_size', type=int, default=16, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=1, help='number of evaluation')
parser.add_argument('--channel', type=int, default=200, help='bandwidth')
parser.add_argument('--patches', type=int, default=9, help='number of patches')
parser.add_argument('--s_patches', type=int, default=9, help='number of patches')
parser.add_argument('--aug_num', type=int, default=1, help='number of augmentations')
parser.add_argument('--sample_num', type=int, default=5, help='K-shot')
parser.add_argument('--d_model', type=int, default=256, help='size of hidden dimension')
parser.add_argument('--out_dims', type=int, default=32, help='size of output dimension')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--epoches', type=int, default=50, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
parser.add_argument('--decay_num', type=int, default=5)
parser.add_argument('--lambdaC', type=float, default=0.01, help='lambda')
parser.add_argument('--temperature', type=float, default=1.0, help='temperature')
parser.add_argument('--itr', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


# -------------------------------------------------------------------------------
# train model
def train_epoch(model, train_loader, criterion, optimizer, num_classes):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):  # 1, B, N, PP & 1, B, N
        batch_data = batch_data.cuda().squeeze(0)  # B, N, PP
        batch_target = batch_target.cuda().squeeze(0)  # B, N

        optimizer.zero_grad()
        batch_pred, batch_repr1, batch_repr2 = model(batch_data, mode='train')
        loss1 = criterion(batch_pred, batch_target)
        loss2 = Contrastive_Loss(batch_repr1, num_classes, args.temperature)
        loss3 = Contrastive_Loss(batch_repr2, num_classes, args.temperature)
        loss = loss1 + args.lambdaC * loss2 + args.lambdaC * loss3
        loss.backward()
        optimizer.step()

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre


# -------------------------------------------------------------------------------
# validate model
def valid_epoch(model, valid_loader, valid_data, support_data, num_classes):
    tar = np.array([])
    pre = np.array([])
    _, support_repr1, support_repr2 = model(support_data.cuda(), mode='train')
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        batch_pred, batch_repr1, batch_repr2 = model(batch_data, mode='train')
        score1 = torch.softmax(batch_pred, dim=-1)
        score2 = Contrastive_Score(batch_repr1, support_repr1, num_classes, args.temperature)
        score3 = Contrastive_Score(batch_repr2, support_repr2, num_classes, args.temperature)
        score = score1 * score2 * score3

        prec1, t, p = accuracy(score, batch_target, topk=(1,))
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return tar, pre


def test_epoch(model, test_loader):
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
        batch_data = batch_data.cuda()
        batch_pred = model(batch_data)

        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())
    return pre


# -------------------------------------------------------------------------------
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA


# -------------------------------------------------------------------------------
# Parameter Setting
np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)
cudnn.deterministic = False  #
cudnn.benchmark = False
# prepare data
if args.dataset == 'Indian':
    data = loadmat('./data/IndianPine.mat')
elif args.dataset == 'Pavia':
    data = loadmat('./data/Pavia.mat')
elif args.dataset == 'Houston':
    data = loadmat('./data/Houston.mat')
elif args.dataset == 'Salinas':
    data = loadmat('./data/Salinas.mat')
else:
    raise ValueError("Unknown dataset")
TR = data['TR']  # (145,145)
TE = data['TE']  # (145,145)
input = data['input']  # (145,145,200)
label = TR + TE
num_classes = np.max(TR)  # 16

# normalize data by band norm
input_normalize = np.zeros(input.shape)
for i in range(input.shape[2]):
    input_max = np.max(input[:, :, i])
    input_min = np.min(input[:, :, i])
    input_normalize[:, :, i] = (input[:, :, i] - input_min) / (input_max - input_min)
# data size
height, width, band = input.shape
logger.info("height={0},width={1},band={2}".format(height, width, band))
# -------------------------------------------------------------------------------
# obtain train and test data
total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = chooose_train_and_test_point(
    TR, TE, label, num_classes)

mirror_image = mirror_hsi(height, width, band, input_normalize, patch=args.patches)
x_train_band, x_test_band = train_and_test_data(mirror_image, band, total_pos_train, total_pos_test,
                                                total_pos_true, patch=args.patches,
                                                s_patch=args.s_patches)
y_train, y_test = train_and_test_label(number_train, number_test, number_true, num_classes)

# -------------------------------------------------------------------------------
# load data
x_train = torch.from_numpy(x_train_band.transpose(0, 2, 1)).type(torch.FloatTensor)  # [695, 200, 7 * 7]
y_train = torch.from_numpy(y_train).type(torch.LongTensor)  # [695]

log_path = './result.log'
logger.info("start training")

logger.info('{}_th iteration'.format(args.itr))
OA, AA_mean, Kappa, AA = 0, 0, 0, 0
V_OA, V_AA_mean, V_Kappa, V_AA = 0, 0, 0, 0

# Create path
path = './checkpoint/' + args.dataset + '_' + str(args.itr)
if not os.path.exists(path):
    os.makedirs(path)
best_model_path = path + '/' + 'checkpoint.pth'  # 最好的模型权重的路径

save_path = './result/' + args.dataset + '_' + str(args.itr)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# create model
model = Network(patch=args.patches, d_model=args.d_model, out_dims=args.out_dims,
                channel=args.channel, class_num=len(number_train), dropout=args.dropout)
model = model.cuda()
# Read Train data

Label_train = Dataset_train(train_data=x_train,  # 训练数据
                            num_train=number_train,  # 每个类的数量
                            train_label=y_train,  # 训练数据标签
                            patch=args.patches,  # 大patch的大小 9 * 9
                            small_patch=args.s_patches,  # 小patch的大小 5 * 5
                            sample_num=args.sample_num,  # N-shot，每个类所用来训练的样本数, 5
                            batch_size=args.batch_size,
                            aug_num=args.aug_num)  # 数据增强的数量
support_data, support_label = Label_train.get_support_set()
support_data = torch.tensor(support_data)
support_label = torch.tensor(support_label)
non_train_index = Label_train.get_non_train_index().reshape(-1).tolist()
# B, N, pp-> 1, B, N, pp
label_train_loader = Data.DataLoader(Label_train, batch_size=1, shuffle=True, drop_last=False)

x_test_band = np.delete(x_test_band, non_train_index, axis=0)  # [695 + 9671 - 80, 200, 9 * 9]
y_test = np.delete(y_test, non_train_index, axis=0)  # [695 + 9671 - 80]

x_test = torch.from_numpy(x_test_band.transpose(0, 2, 1)).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test).type(torch.LongTensor)
Label_test = Data.TensorDataset(x_test, y_test)

label_test_loader = Data.DataLoader(Label_test, batch_size=args.batch_size, shuffle=False, drop_last=True)

# criterion
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=args.epoches // args.decay_num, gamma=args.gamma)
tic = time.time()
pred_list = []

for epoch in range(args.epoches):
    # train model
    OA_C, AA_mean_C, Kappa_C, AA_C = 0, 0, 0, 0
    model.train()
    train_acc, train_obj, tar_t, pre_t = train_epoch(model, label_train_loader, criterion, optimizer,
                                                     len(number_train))
    scheduler.step()

    if epoch % args.test_freq == 0:
        model.eval()
        tar_v, pre_v = valid_epoch(model, label_test_loader, Label_test,
                                   support_data, len(number_train))

        OA_C, AA_mean_C, Kappa_C, AA_C = output_metric(tar_v, pre_v)

        if OA_C > OA:
            OA, AA_mean, Kappa, AA = OA_C, AA_mean_C, Kappa_C, AA_C
            torch.save(model.state_dict(), best_model_path)

        if epoch > args.epoches - 10:
            pred_list.append(pre_v)
        if epoch == args.epoches - 1:
            pred_list = np.stack(pred_list, axis=0)
            vote_pred_v = np.zeros(pre_v.shape)
            for i in range(pre_v.shape[0]):
                instance_list = pred_list[:, i].tolist()
                Count_list = Counter(instance_list)
                vote_pred_v[i] = Count_list.most_common()[0][0]
            V_OA, V_AA_mean, V_Kappa, V_AA = output_metric(tar_v, vote_pred_v)

    logger.info("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f} OA_best: {:.4f} OA_Current: {:.4f}"
                .format(epoch + 1, train_obj, train_acc, OA, OA_C))

logger.info("{}_itr's result: ".format(str(args.itr)))
logger.info("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA, AA_mean, Kappa))
logger.info("Vote | OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(V_OA, V_AA_mean, V_Kappa))

np.save(save_path + '/' + f'OA.npy', OA)
np.save(save_path + '/' + f'AA_mean.npy', AA_mean)
np.save(save_path + '/' + f'Kappa.npy', Kappa)
np.save(save_path + '/' + f'AA.npy', AA)
np.save(save_path + '/' + f'V_OA.npy', V_OA)
np.save(save_path + '/' + f'V_AA_mean.npy', V_AA_mean)
np.save(save_path + '/' + f'V_Kappa.npy', V_Kappa)
np.save(save_path + '/' + f'V_AA.npy', V_AA)
with open(log_path, "a") as f:
    f.write(time.strftime("%Y-%m-%d-%H_%M_%S || ", time.localtime()))
    f.write("{}_itr's result: ".format(str(args.itr)))
    f.write("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f} || ".format(OA, AA_mean, Kappa))
    f.write("V_OA: {:.4f} | V_AA: {:.4f} | V_Kappa: {:.4f}".format(V_OA, V_AA_mean, V_Kappa) + '\n')
    f.flush()
    f.close()

logger.info("**************************************************")

toc = time.time()
logger.info("Running Time: {:.2f}".format(toc - tic))
logger.info("**************************************************")
torch.cuda.empty_cache()
