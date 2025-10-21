import argparse
import os
import numpy as np
import shutil
from loguru import logger
logger.add(
    './info.log',
    format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
    level="INFO",
)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--save', action='store_true', help='whether saving results and checkpoints', default=False)
parser.add_argument('--dataset', choices=['Indian', 'Pavia', 'Houston', 'Salinas'], default='Indian', help='dataset to use')
args = parser.parse_args()

log_path = './result.log'

# Best
OA_f, AA_mean_f, Kappa_f, AA_f = [], [], [], []
for i in range(1, 11):
    save_path = './result/' + args.dataset + '_' + str(i)
    OA_f.append(np.load(save_path + '/' + f'OA.npy'))
    AA_f.append(np.load(save_path + '/' + f'AA.npy'))
    Kappa_f.append(np.load(save_path + '/' + f'Kappa.npy'))
    AA_mean_f.append(np.load(save_path + '/' + f'AA_mean.npy'))

mean_OA = np.mean(OA_f)
std_OA = np.std(OA_f)

AA_f = np.stack(AA_f, axis=0)
mean_AA_f = np.mean(AA_f, axis=0)
std_AA_f = np.std(AA_f, axis=0)

mean_AA = np.mean(AA_mean_f)
std_AA = np.std(AA_mean_f)
mean_Kappa = np.mean(Kappa_f)
std_Kappa = np.std(Kappa_f)
logger.info("Best final result:")
logger.info("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(mean_OA, mean_AA, mean_Kappa))
with open(log_path, "a") as f:
    f.write("Best final result:" + '\n')
    f.write("OA: {0:.4f} ± {1:.4f}".format(mean_OA, std_OA) + '\n')
    f.write("AA: {0:.4f} ± {1:.4f}".format(mean_AA, std_AA) + '\n')
    f.write("Kappa: {0:.4f} ± {1:.4f}".format(mean_Kappa, std_Kappa) + '\n')
    f.write("Category-wise Accuracy: " + '\n')
    for i in range(len(mean_AA_f)):
        f.write("Category {0:.0f}: {1:.4f} ± {2:.4f}".
                format(i, mean_AA_f[i], std_AA_f[i]) + '\n')
    f.flush()
    f.close()

# Vote
OA_f, AA_mean_f, Kappa_f, AA_f = [], [], [], []
for i in range(1, 11):
    save_path = './result/' + args.dataset + '_' + str(i)
    OA_f.append(np.load(save_path + '/' + f'V_OA.npy'))
    AA_f.append(np.load(save_path + '/' + f'V_AA.npy'))
    Kappa_f.append(np.load(save_path + '/' + f'V_Kappa.npy'))
    AA_mean_f.append(np.load(save_path + '/' + f'V_AA_mean.npy'))

mean_OA = np.mean(OA_f)
std_OA = np.std(OA_f)

AA_f = np.stack(AA_f, axis=0)
mean_AA_f = np.mean(AA_f, axis=0)
std_AA_f = np.std(AA_f, axis=0)

mean_AA = np.mean(AA_mean_f)
std_AA = np.std(AA_mean_f)
mean_Kappa = np.mean(Kappa_f)
std_Kappa = np.std(Kappa_f)
logger.info("Voted final result:")
logger.info("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(mean_OA, mean_AA, mean_Kappa))
with open(log_path, "a") as f:
    f.write("Voted final result:" + '\n')
    f.write("OA: {0:.4f} ± {1:.4f}".format(mean_OA, std_OA) + '\n')
    f.write("AA: {0:.4f} ± {1:.4f}".format(mean_AA, std_AA) + '\n')
    f.write("Kappa: {0:.4f} ± {1:.4f}".format(mean_Kappa, std_Kappa) + '\n')
    f.write("Category-wise Accuracy: " + '\n')
    for i in range(len(mean_AA_f)):
        f.write("Category {0:.0f}: {1:.4f} ± {2:.4f}".
                format(i, mean_AA_f[i], std_AA_f[i]) + '\n')
    f.flush()
    f.close()

if not args.save:
    for ii in range(1, 11):
        weight_path = './checkpoint/' + args.dataset + '_' + str(ii)
        if os.path.exists(weight_path):
            shutil.rmtree(weight_path)

        save_path = './result/' + args.dataset + '_' + str(ii)
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
