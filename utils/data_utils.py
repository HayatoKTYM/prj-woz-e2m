"""
Dataset 構築するプログラム
"""

import pandas as pd
import numpy as np
import torch
import cv2
from collections import Counter

class MakeTargetDataset(object):

    def __init__(self, sp_path_list, label_path_list):

        self.sp_path_list = sp_path_list
        self.label_path_list = label_path_list

    def __call__(self, start, end,reset_flag=False):
        x_spA = x_spB = y1 = y2 = []
        x_img = y3 = []
        x_t = y = []

        for sp_f, f in zip(self.sp_path_list[start:end], self.label_path_list[start:end]):

            df = pd.read_csv(sp_f)
            x_spA = np.append(x_spA, df.iloc[:-1,:256].values, axis=0) \
                        if len(x_spA) != 0 else df.iloc[:, :256].values[:-1]
            x_spB = np.append(x_spB, df.iloc[:-1, 256:].values, axis=0) \
                if len(x_spB) != 0 else df.iloc[:, 256:].values[:-1]
            length = len(df)

            df = pd.read_csv(f)[:length]
            #y1 = np.append(y1, df['utter_A'].values[:length - 1]) if len(y1) != 0 else df['utter_A'].values[:length - 1]
            #y2 = np.append(y2, df['utter_B'].values[:length - 1]) if len(y2) != 0 else df['utter_B'].values[:length - 1]
            #y3 = np.append(y3, df['gaze'].values[:length - 1]) if len(y3) != 0 else df['gaze'].values[:length - 1]
            x_img = np.append(x_img, load_image(df['path'].values[:-1]), axis=0) \
                if len(x_img) != 0 else load_image(df['path'].values[:-1])
            label = df['target'].map(lambda x: 0 if x == 'A' else 1).values
            x_t = np.append(x_t, label[:-1], axis=0) if len(x_t) != 0 else label[:-1]
            y = np.append(y, label[1:]) if len(y) != 0 else label[1:]

            if reset_flag:
                """
                会話データ間にreset flagを埋め込むか
                LSTMを用いた stateful model を実装するなら必要
                """
                x_spA = np.append(x_spA, np.zeros((1,256)), axis=0)
                x_spB = np.append(x_spB, np.zeros((1,256)), axis=0)
                x_img = np.append(x_img, np.zeros((1,1,32,96)), axis=0)
                x_t = np.append(x_t, 0)
                y = np.append(y,-1)

        print(x_spA.shape, x_img.shape, y.shape, x_t.shape)

        return torch.tensor(x_spA, dtype=torch.float32), torch.tensor(x_spB, dtype=torch.float32), \
               torch.tensor(x_img, dtype=torch.float32), torch.tensor(x_t, dtype=torch.float32), \
               torch.tensor(y, dtype=torch.long)

class MakeTargetLLDataset(object):

    def __init__(self, sp_path_list, label_path_list):

        self.sp_path_list = sp_path_list
        self.label_path_list = label_path_list

    def __call__(self, start, end,reset_flag=False):
        x_spA = x_spB = y1 = y2 = []
        x_img = y3 = []
        x_t = y = []
        sp_step = 10
        for sp_f, f in zip(self.sp_path_list[start:end], self.label_path_list[start:end]):

            df = pd.read_csv(sp_f)
            df = df[:len(df) // sp_step * sp_step]
            x_spA = np.append(x_spA, df.iloc[:,:114].values, axis=0) \
                        if len(x_spA) != 0 else df.iloc[:, :114].values
            x_spB = np.append(x_spB, df.iloc[:, 114:].values, axis=0) \
                if len(x_spB) != 0 else df.iloc[:, 114:].values
            length = len(df) // sp_step

            df = pd.read_csv(f)[:length]
            #y1 = np.append(y1, df['utter_A'].values[:length - 1]) if len(y1) != 0 else df['utter_A'].values[:length - 1]
            #y2 = np.append(y2, df['utter_B'].values[:length - 1]) if len(y2) != 0 else df['utter_B'].values[:length - 1]
            #y3 = np.append(y3, df['gaze'].values[:length - 1]) if len(y3) != 0 else df['gaze'].values[:length - 1]
            x_img = np.append(x_img, load_image(df['path'].values[:-1]), axis=0) \
                if len(x_img) != 0 else load_image(df['path'].values[:-1])
            label = df['target'].map(lambda x: 0 if x == 'A' else 1).values
            x_t = np.append(x_t, label[:-1], axis=0) if len(x_t) != 0 else label[:-1]
            y = np.append(y, label[1:]) if len(y) != 0 else label[1:]

            if reset_flag:
                """
                会話データ間にreset flagを埋め込むか
                LSTMを用いた stateful model を実装するなら必要
                """
                x_spA = np.append(x_spA, np.zeros((sp_step,114)), axis=0)
                x_spB = np.append(x_spB, np.zeros((sp_step,114)), axis=0)
                x_img = np.append(x_img, np.zeros((1,1,32,96)), axis=0)
                x_t = np.append(x_t, 0)
                y = np.append(y,-1)

        x_spA = x_spA.reshape(-1, sp_step, 114)
        x_spB = x_spB.reshape(-1, sp_step, 114)
        print(x_spA.shape, x_img.shape, y.shape, x_t.shape)

        return torch.tensor(x_spA, dtype=torch.float32), torch.tensor(x_spB, dtype=torch.float32), \
               torch.tensor(x_img, dtype=torch.float32), torch.tensor(x_t, dtype=torch.float32), \
               torch.tensor(y, dtype=torch.long)

class MakeActionDataset(object):

    def __init__(self, sp_path_list, label_path_list):

        self.sp_path_list = sp_path_list
        self.label_path_list = label_path_list

    def __call__(self, start, end, reset_flag=False):
        x_spA = x_spB = y1 = y2 = []
        x_img = y3 = []
        x_t = y = []

        for sp_f, f in zip(self.sp_path_list[start:end], self.label_path_list[start:end]):

            df = pd.read_csv(sp_f)
            x_spA = np.append(x_spA, df.iloc[:-1,:256].values, axis=0) \
                        if len(x_spA) != 0 else df.iloc[:, :256].values[:-1]
            x_spB = np.append(x_spB, df.iloc[:-1, 256:].values, axis=0) \
                if len(x_spB) != 0 else df.iloc[:, 256:].values[:-1]
            length = len(df)

            df = pd.read_csv(f)[:length]
            #y1 = np.append(y1, df['utter_A'].values[:length - 1]) if len(y1) != 0 else df['utter_A'].values[:length - 1]
            #y2 = np.append(y2, df['utter_B'].values[:length - 1]) if len(y2) != 0 else df['utter_B'].values[:length - 1]
            #y3 = np.append(y3, df['gaze'].values[:length - 1]) if len(y3) != 0 else df['gaze'].values[:length - 1]
            x_img = np.append(x_img, load_image(df['path'].values[:-1]), axis=0) \
                if len(x_img) != 0 else load_image(df['path'].values[:-1])
            label = df['target'].map(lambda x: 0 if x == 'A' else 1).values
            x_t = np.append(x_t, label[:-1], axis=0) if len(x_t) != 0 else label[:-1]
            label = df['action'].map(lambda x: 1 if x == 'Passive' else 0).values
            label = add_positive_label(label)
            y = np.append(y, label[1:]) if len(y) != 0 else label[1:]

            if reset_flag:
                """
                会話データ間にreset flagを埋め込むか
                LSTMを用いた stateful model を実装するなら必要
                """
                x_spA = np.append(x_spA, np.zeros((1,256)), axis=0)
                x_spB = np.append(x_spB, np.zeros((1,256)), axis=0)
                x_img = np.append(x_img, np.zeros((1,1,32,96)), axis=0)
                x_t = np.append(x_t, 0)
                y = np.append(y,-1)

        print(x_spA.shape, x_img.shape, y.shape, x_t.shape)
        print(Counter(y))

        return torch.tensor(x_spA, dtype=torch.float32), torch.tensor(x_spB, dtype=torch.float32), \
               torch.tensor(x_img, dtype=torch.float32), torch.tensor(x_t, dtype=torch.float32), \
               torch.tensor(y, dtype=torch.long)

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

def add_positive_label(y):
    y_after = [0]*len(y)
    y_after = y.copy()
    for i in range(len(y)):
        if y[i] != 1: continue
        for j in range(max(i-5,0),min(len(y),i+5)):
            y_after[j] = 1
    return y_after
