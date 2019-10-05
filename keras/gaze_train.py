__author__ = 'Hayato Katayama'
"""
視線推定プログラム
label 0 look robot
        1 not look robot

GazeTrain関数 　　　このプログラムを実行する時に使用
TimeGazeTrain関数　アクション推定など複数画像を系列として処理したい時に使用(import用)

"""
from keras_model import *
import keras

import glob
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import KFold

def load_image(paths,gray_flag=0):
    #pathを受け取って画像を返す
    img_feature = []
    for path in paths:
        x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if x is None:
            x = np.array([0]*32*96)
        x = x.reshape(32,96,1)
        img_feature.append(x / 255.0)
    return np.array(img_feature,dtype=np.float32)

def extract_img(csv_f :str = ''):
    """
    param: csvファイル
    return 画像配列(None,1,32,96), ラベル(None,)
    """
    df = pd.read_csv(csv_f)
    label = df.iloc[:,1].values
    feature = df.iloc[:,2:].values
    assert feature.shape == (len(feature),32*96), print('data load error')
    return feature/255.0,label

def shuffle_samples(X, y):
    """
    X,y をindexの対応を崩さずにshuffleして返す関数
    param: X,y
    return X,y
    """
    assert len(X) == len(y), print('data length inccorrect')
    zipped = list(zip(X, y))
    np.random.shuffle(zipped)
    X_result, y_result = zip(*zipped)
    return np.array(X_result,dtype=np.float32), np.array(y_result,dtype=np.int32)    # 型をnp.arrayに変換

if __name__ == '__main__':

    eye_files = sorted(glob.glob('/mnt/aoni02/katayama/short_project/proken2018_A/input/*')) #人ごとの画像フォルダ
    val_cnt = 0
    kf = KFold(n_splits=10)
    test_Acc = []
    for train_index, test_index in kf.split(eye_files[:10]):
        X_train,X_val,X_test=[],[],[]
        y_train,y_val,y_test=[],[],[]
        print(train_index,test_index)
        for i in train_index[:-3]:
            feature, label = extract_img(eye_files[i]+'/no_eye_data.csv')
            X_train.extend(feature)
            y_train.extend(label)
            feature, label = extract_img(eye_files[i]+'/yes_eye_data.csv')
            X_train.extend(feature)
            y_train.extend(label)
        for i in train_index[-3:]:
            feature, label = extract_img(eye_files[i]+'/no_eye_data.csv')
            X_val.extend(feature)
            y_val.extend(label)
            feature, label = extract_img(eye_files[i]+'/yes_eye_data.csv')
            X_val.extend(feature)
            y_val.extend(label)
        for i in test_index:
            feature, label = extract_img(eye_files[i]+'/no_eye_data.csv')
            X_test.extend(feature)
            y_test.extend(label)
            feature, label = extract_img(eye_files[i]+'/yes_eye_data.csv')
            X_test.extend(feature)
            y_test.extend(label)


        X_train = np.array(X_train,dtype=np.float32).reshape(-1,32,96,1)
        X_val = np.array(X_val,dtype=np.float32).reshape(-1,32,96,1)
        X_test = np.array(X_test,dtype=np.float32).reshape(-1,32,96,1)
        y_train = np.array(y_train,dtype=np.int32)
        y_val = np.array(y_val,dtype=np.int32)
        y_test = np.array(y_test,dtype=np.int32)  
        X_train,y_train = shuffle_samples(X_train,y_train)
        print('training size:',len(X_train),'validation size:',len(X_val),'test size:',len(X_test))

        #Training phase ##
        model = GazeTrain()
        #model.summary()
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,verbose=1)
        model_save = keras.callbacks.ModelCheckpoint(filepath="result/gaze/weights.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.h5", monitor='val_loss',save_weights_only=True)
        hist = model.fit(X_train,y_train,
              epochs=50,
              batch_size=128,
              callbacks = [
                               early_stopping,
                               #model_save
                              ],
              verbose=0,
              validation_data = (X_val,y_val),
              validation_split=0.25)
        print('Accuracy:',model.evaluate(X_test,y_test)[1])
