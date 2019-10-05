__author__ = 'Hayato Katayama'
__date__ = '20190927'

"""
発話推定を行うプログラム
入力 LLD特徴量
出力 0:発話していない / 1:発話中
"""

import pandas as pd
import numpy as np
import torch
import cv2
import torch.optim as optim
import torch.nn as nn
import glob
from collections import Counter

sys.path.append('../model/')
from cnn_model import VADPredictLLD


def train(net, dataloaders_dict, criterion, optimizer, num_epochs):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    net.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')
        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()   # モデルを検証モードに

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数

            if (epoch == 0) and (phase == 'train'): # 未学習時の検証性能を確かめるため、epoch=0の訓練は省略
                continue
            cnt=0
            for inputs, labels in dataloaders_dict[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                if cnt == 0:
                    out = net(inputs)
                    #out, h, c = net(inputs,None,None) # 順伝播
                    cnt = 1
                else:
                    out = net(inputs)
                    #out, h, c = net(inputs,h.detach(),c.detach()) # 順伝播

                loss = criterion(out, labels) # ロスの計算
                _, preds = torch.max(out, 1)  # ラベルを予測

                if phase == 'train': # 訓練時はバックプロパゲーション
                    optimizer.zero_grad() # 勾配の初期化
                    loss.backward()#retain_graph=True) # 勾配の計算
                    optimizer.step()# パラメータの更新

                epoch_loss += loss.item() * inputs.size(0)  # lossの合計を更新
                epoch_corrects += torch.sum(preds == labels.data) # 正解数の合計を更新

            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('')

    y_true, y_pred = np.array([]), np.array([])
    for inputs, labels in dataloaders_dict['test']:
        inputs = inputs.to(device)

        out = net(inputs) # 順伝播
        #loss = criterion(out, labels) # ロスの計算
        _, preds = torch.max(out, 1)  # ラベルを予測

        y_true = np.append(y_true, labels.data.numpy())
        y_pred = np.append(y_pred, preds.cpu().data.numpy())

    from sklearn.metrics import accuracy_score , confusion_matrix
    print(accuracy_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    torch.save(net.state_dict(), './vad_lld_model.pth')


def main():
    #画像pathの読み込み
    sp_files = sorted(glob.glob('/mnt/aoni02/katayama/dataset/DATA2019/LLD/*csv'))
    feature_files = sorted(glob.glob('/mnt/aoni02/katayama/dataset/DATA2019/feature/*csv'))
    sp_step = 10
    x = y = Acc = []

    for sp_f, f in zip(sp_files[:], feature_files[:]):
        df = pd.read_csv(sp_f)

        df = df[:len(df) // sp_step * sp_step]
        x = np.append(x, df.iloc[:, :114].values, axis=0) if len(x) != 0 else df.iloc[:, :114].values
        x = np.append(x, df.iloc[:, 114:].values, axis=0) if len(x) != 0 else df.iloc[:, 114:].values

        length = len(df) // sp_step

        df = pd.read_csv(f)[:length ]

        assert len(df) >= length, print(sp_f)
        y = np.append(y, df['utter_A'].values[1:]) if len(y) != 0 else df['utter_A'].values[1:]
        y = np.append(y, df['utter_B'].values[1:]) if len(y) != 0 else df['utter_B'].values
        x = x[:sp_step * len(y)]
    print(Counter(y))
    x = x.reshape(-1, sp_step, 114)
    x = torch.tensor(x,dtype=torch.float32)
    y = torch.tensor(y,dtype=torch.long)
    print(x.shape,y.shape)

    batch_size = 128
    length = len(y) // 4 * 3
    train_dataloader = torch.utils.data.DataLoader(
        list(zip(x[:length],y[:])), batch_size=batch_size, shuffle=True)

    val_dataloader = torch.utils.data.DataLoader(
        list(zip(x[length:], y[length:])), batch_size=batch_size, shuffle=False)

    test_dataloader = torch.utils.data.DataLoader(
        list(zip(x[length:], y[length:])), batch_size=batch_size, shuffle=False)

    # 辞書オブジェクトにまとめる
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}

    net=VADPredictLLD()
    print(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-03)

    for name, param in net.named_parameters():
        if 'fc' in name or 'lstm' in name:
            param.requires_grad = True
            print("勾配計算あり。学習する：", name)
        else:
            param.requires_grad = False

    train(net, dataloaders_dict, criterion, optimizer, num_epochs=50)

if __name__ == '__main__':
    main()
