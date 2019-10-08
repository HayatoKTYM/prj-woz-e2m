__author__ = 'Hayato Katayama'
__date__ = '20190927'

"""
視線推定を行うプログラム
"""
import sys
sys.path.append('../model/')
from cnn_model import GazeTrain

import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import glob
import cv2
from collections import Counter
from sklearn.model_selection import KFold

def load_image(paths,gray_flag=0):
    #pathを受け取って画像を返す
    img_feature = []
    for path in paths:
        x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if x is None:
            x = np.array([0]*32*96)
        x = x.reshape(1,32,96)
        img_feature.append(x / 255.0)
    return np.array(img_feature,dtype=np.float32)

def extract_img(folder_path :str = '', label : str = 'yes'):
    """
    param: folder path
    param: label (yes[look]  or no[not look])
    return 画像配列(None,96,96,1), ラベル(None,)
    """
    img_paths = glob.glob(folder_path + '/*png')
    img_feature = load_image(img_paths)
    label = np.zeros(len(img_feature)) if label == 'yes' else np.ones(len(img_feature))
    return img_feature, label



def train(net, dataloaders_dict, criterion, optimizer, num_epochs):
    # 学習ループ
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
            for inputs, labels in dataloaders_dict[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                out  = net(inputs)  # 順伝播

                loss = criterion(out, labels)  # ロスの計算
                _, preds = torch.max(out, 1)  # ラベルを予測

                if phase == 'train':  # 訓練時はバックプロパゲーション
                    optimizer.zero_grad()  # 勾配の初期化
                    loss.backward()  # retain_graph=True) # 勾配の計算
                    optimizer.step()  # パラメータの更新

                epoch_loss += loss.item() * inputs.size(0)  # lossの合計を更新
                epoch_corrects += torch.sum(preds == labels.data)  # 正解数の合計を更新

            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('')

    y_true, y_pred = np.array([]), np.array([])
    for inputs, labels in dataloaders_dict['test']:
        inputs = inputs.to(device)

        out = net(inputs)  # 順伝播
        # loss = criterion(out, labels) # ロスの計算
        _, preds = torch.max(out, 1)  # ラベルを予測

        y_true = np.append(y_true, labels.data.numpy())
        y_pred = np.append(y_pred, preds.cpu().data.numpy())

    from sklearn.metrics import accuracy_score, confusion_matrix
    print(accuracy_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    #torch.save(net.state_dict(), './gaze_model.pth')

def main():
    # 画像pathの読み込み
    folder = '/mnt/aoni02/katayama/short_project/proken2018_A/data/eye/'
    yes_face_files = sorted(glob.glob(folder + 'yes/*'))[:]  # 人ごとの画像フォルダ
    no_face_files = sorted(glob.glob(folder + 'no/*'))[:]  # 人ごとの画像フォルダ
    x = y = Acc = []
    val_cnt = 0
    kf = KFold(n_splits=10)
    test_Acc = []

    for train_index, test_index in kf.split(yes_face_files[:]):
        X_train, X_val, X_test = [], [], []
        y_train, y_val, y_test = [], [], []
        print(train_index, test_index)
        print('## making training dataset ##')
        for i in train_index[:-3]:
            try:
                feature, label = extract_img(yes_face_files[i], label='yes')
                X_train = np.append(X_train, feature, axis=0) if len(X_train) != 0 else feature
                y_train = np.append(y_train, label)
                feature, label = extract_img(no_face_files[i], label='no')
                X_train = np.append(X_train, feature, axis=0) if len(X_train) != 0 else feature
                y_train = np.append(y_train, label)
            except:
                print('data not found , maybe there is no path in ' + no_face_files[i])
                continue
        print('## making validation dataset ##')
        for i in train_index[-3:]:
            try:
                feature, label = extract_img(yes_face_files[i], label='yes')
                X_val = np.append(X_val, feature, axis=0) if len(X_val) != 0 else feature
                y_val = np.append(y_val, label)
                feature, label = extract_img(no_face_files[i], label='no')
                X_val = np.append(X_val, feature, axis=0) if len(X_val) != 0 else feature
                y_val = np.append(y_val, label)
            except:
                print('data not found , maybe there is no path in ' + no_face_files[i])
                continue
        print('## making test dataset ##')
        for i in test_index:
            try:
                feature, label = extract_img(yes_face_files[i], label='yes')
                X_test = np.append(X_test, feature, axis=0) if len(X_test) != 0 else feature
                y_test = np.append(y_test, label)
                feature, label = extract_img(no_face_files[i], label='no')
                X_test = np.append(X_test, feature, axis=0) if len(X_test) != 0 else feature
                y_test = np.append(y_test, label)
            except:
                print('data not found , maybe there is no path in ' + no_face_files[i])
                continue

        print('training size:', len(X_train), 'validation size:', len(X_val), 'test size:', len(X_test))
        print('label num ', Counter(y_train))

        x = torch.tensor(X_train, dtype=torch.float32)
        y = torch.tensor(y_train, dtype=torch.long)

        batch_size = 64
        train_dataloader = torch.utils.data.DataLoader(
            list(zip(x,y)), batch_size=batch_size, shuffle=True)

        x = torch.tensor(X_val, dtype=torch.float32)
        y = torch.tensor(y_val, dtype=torch.long)
        val_dataloader = torch.utils.data.DataLoader(
            list(zip(x,y)), batch_size=batch_size, shuffle=False)

        x = torch.tensor(X_test, dtype=torch.float32)
        y = torch.tensor(y_test, dtype=torch.long)
        test_dataloader = torch.utils.data.DataLoader(
            list(zip(x,y)), batch_size=batch_size, shuffle=False)

        # 辞書オブジェクトにまとめる
        dataloaders_dict = {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}

        net = GazeTrain()
        print(net)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.1)
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        train(net=net, dataloaders_dict=dataloaders_dict, criterion=criterion, optimizer=optimizer, num_epochs=50)

        break


if __name__ == '__main__':
    main()
