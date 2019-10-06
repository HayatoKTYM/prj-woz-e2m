__author__ = 'Hayato Katayama'
__date__ = '20190927'

"""
ロボットの顔向き(target)推定を行うプログラム
"""
import sys
sys.path.append('../utils/')
from data_utils import MakeTargetLLDataset
sys.path.append('../model/')
from cnn_model import *
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import glob


def eval(net, dataloader):
    # 学習ループ
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    net.to(device)

    y_true, y_pred = np.array([]), np.array([])
    cnt = 0
    for inputs,inputs2,inputs3,inputs4, labels in dataloader:
        inputs = inputs.to(device)
        inputs2 = inputs2.to(device)
        inputs3 = inputs3.to(device)
        inputs4 = inputs4.to(device)
        if labels.data.numpy()[0] == -1:
            h, c = net.reset_state(1)
            #print('reset state! conversation changed!')
            continue
        if cnt == 0:
            out, h, c = net(inputs,inputs2,inputs3,inputs4, None, None)
            cnt += 1
        else:
            out, h, c = net(inputs,inputs2,inputs3,torch.tensor(preds,dtype=torch.float).to(device), h.detach(), c.detach())  # 順伝播
        # loss = criterion(out, labels) # ロスの計算
        _, preds = torch.max(out, 1)  # ラベルを予測

        y_true = np.append(y_true, labels.data.numpy())
        y_pred = np.append(y_pred, preds.cpu().data.numpy())

    from sklearn.metrics import accuracy_score, confusion_matrix
    print(accuracy_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))

def main():

    sp_files = sorted(glob.glob('/mnt/aoni02/katayama/dataset/DATA2019/LLD/*csv'))
    feature_files = sorted(glob.glob('/mnt/aoni02/katayama/dataset/DATA2019/feature/*csv'))

    datamaker = MakeTargetLLDataset(sp_files, feature_files)
    x_spA_test, x_spB_test, x_img_test, x_t_test, y_test = datamaker(start=48, end=53,reset_flag=True)

    batch_size = 1
    test_dataloader = torch.utils.data.DataLoader(
        list(zip(x_img_test, x_spA_test, x_spB_test, x_t_test, y_test)), batch_size=batch_size, shuffle=False)

    net = TargetPredictLLD()
    print(net)
    net.load_state_dict(torch.load('../result/target/target_lld_model.pth'))

    eval(net, test_dataloader)


if __name__ == '__main__':
    main()
