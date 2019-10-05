__author__ = 'Hayato Katayama'
__date__ = '20190927'

"""
ロボットの応答義務推定を行うプログラム
"""
import sys
sys.path.append('../utils')
from data_utils import *
sys.path.append('../model/')
from cnn_model import *
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import glob
from collections import Counter

def train(net, dataloaders_dict, criterion, optimizer, num_epochs):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print(device)
    net.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-------------')
        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()  # モデルを検証モードに

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数

            if (epoch == 0) and (phase == 'train'):  # 未学習時の検証性能を確かめるため、epoch=0の訓練は省略
                continue
            cnt = 0
            y_past = torch.tensor([0.0]) #１時刻前の予測
            for inputs,inputs2,inputs3,inputs4, labels in dataloaders_dict[phase]:
                inputs = inputs.to(device)
                inputs2 = inputs2.to(device)
                inputs3 = inputs3.to(device)
                inputs4 = inputs4.to(device)
                if labels.data.numpy()[0] == -1:
                    h, c = net.reset_state(1)
                    y_past = torch.tensor([0.0])
                    #print('reset state! conversation changed!')
                    continue
                labels = labels.to(device)
                if cnt == 0 :
                    out, h, c = net(inputs,inputs2,inputs3,inputs4, None, None, y_past.detach())  # 順伝播
                    cnt += 1
                else:
                    out, h, c = net(inputs,inputs2,inputs3,inputs4, h, c, y_past.detach())  # 順伝播
                    cnt += 1
                loss = criterion(out, labels) if cnt % 50 == 1 else loss + criterion(out, labels)  # ロスの計算
                _, preds = torch.max(out, 1)  # ラベルを予測

                if cnt % 128 == 0:
                    # 50会入力したら纏めて誤差逆伝播
                    if phase == 'train':  # 訓練時はバックプロパゲーション
                        optimizer.zero_grad()  # 勾配の初期化
                        loss.backward()  # retain_graph=True) # 勾配の計算
                        optimizer.step()  # パラメータの更新
                        h,c = h.detach(), c.detach()
                    epoch_loss += loss.item() * inputs.size(0)  # lossの合計を更新
                y_past = criterion(out,labels)
                epoch_corrects += torch.sum(preds == labels.data)  # 正解数の合計を更新

            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('')

    y_true, y_pred = np.array([]), np.array([])
    cnt = 0
    y_past = torch.tensor([0.0]) #１時刻前の予測
    for inputs,inputs2,inputs3,inputs4, labels in dataloaders_dict['test']:
        inputs = inputs.to(device)
        inputs2 = inputs2.to(device)
        inputs3 = inputs3.to(device)
        inputs4 = inputs4.to(device)
        if labels.data.numpy()[0] == -1:
            h, c = net.reset_state(1)
            y_past = torch.tensor([0.0]) #１時刻前の予測
            print('reset state! conversation changed!')
            continue
        if cnt == 0:
            out, h, c = net(inputs,inputs2,inputs3,inputs4, None, None,y_past.detach())
            cnt += 1
        else:
            out, h, c = net(inputs,inputs2,inputs3,inputs4, h.detach(), c.detach(), y_past.detach())  # 順伝播
        # loss = criterion(out, labels) # ロスの計算
        _, preds = torch.max(out, 1)  # ラベルを予測
        y_past = criterion(out, labels)
        y_true = np.append(y_true, labels.data.numpy())
        y_pred = np.append(y_pred, preds.cpu().data.numpy())

    from sklearn.metrics import accuracy_score, confusion_matrix
    print(accuracy_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    torch.save(net.state_dict(), './response_model_13_83.pth')

def main():
    # 画像pathの読み込み
    sp_files = sorted(glob.glob('/mnt/aoni02/katayama/dataset/DATA2019/sp/*csv'))
    feature_files = sorted(glob.glob('/mnt/aoni02/katayama/dataset/DATA2019/feature/*csv'))

    datamaker = MakeActionDataset(sp_files, feature_files)
    x_spA, x_spB, x_img, x_t, y = datamaker(start=13,end=83)
    x_spA_val, x_spB_val, x_img_val, x_t_val, y_val = datamaker(start=83, end=87)
    x_spA_test, x_spB_test, x_img_test, x_t_test, y_test = datamaker(start=83, end=91)

    batch_size = 1
    train_dataloader = torch.utils.data.DataLoader(
        list(zip(x_img,x_spA,x_spB,x_t,y)), batch_size=batch_size, shuffle=False)

    val_dataloader = torch.utils.data.DataLoader(
        list(zip(x_img_val, x_spA_val, x_spB_val, x_t_val, y_val)), batch_size=batch_size, shuffle=False)

    test_dataloader = torch.utils.data.DataLoader(
        list(zip(x_img_test, x_spA_test, x_spB_test, x_t_test, y_test)), batch_size=batch_size, shuffle=False)

    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}

    net = ResponsePredict(num_layers=2)
    print(net)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device='cpu'
    weights = torch.tensor([1.0, 3.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights,reduce=False)
    #optimizer = optim.SGD(net.parameters(), lr=0.1)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for name, param in net.named_parameters():
        if 'gaze' not in name and 'vad' not in name:
            param.requires_grad = True
            print("勾配計算あり。学習する：", name)
        else:
            param.requires_grad = False
            # print("勾配計算なし。学習しない：", name)

    train(net=net, dataloaders_dict=dataloaders_dict, criterion=criterion, optimizer=optimizer, num_epochs=10)


if __name__ == '__main__':
    main()
