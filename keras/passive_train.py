from keras_model import *
import keras
from NLP import *

import glob
import pandas as pd
import cv2
import numpy as np
from collections import Counter
import argparse

def load_image(paths,gray_flag=0,timestep=10):
    #pathを受け取って画像を返す
    img_feature = []
    for path in paths:
        x = cv2.imread(path.replace('eye','face'), cv2.IMREAD_GRAYSCALE)
        if x is None:
            x = np.array([0]*96*96)
        x = x.reshape(96,96,1)
        img_feature.append(x / 255.0)
    return np.array([img_feature[i:i+timestep] for i in range(len(paths)-timestep)],dtype=np.float32)

def add_positive_label(y):
    y_after = [0]*len(y)
    y_after = y.copy()
    for i in range(len(y)):
        if y[i] != 1: continue
        for j in range(max(i-5,0),min(len(y),i+5)):
            y_after[j] = 1
    return y_after

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each omini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result/passive/train13-60/',
                        help='Directory to output the result')
    parser.add_argument('--resume_model', type=str)
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--stop_trigger', type=int, default=10)
    parser.add_argument('--input1', type=int, default=13)
    parser.add_argument('--input2', type=int, default=60)
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')
    print(f'stop_epoch: {args.stop_trigger}')
    print('')
    print('#training and validation using ',args.input1,' ~ ',args.input2,' conversations')
    print('')
    
    sp_files = sorted(glob.glob('/mnt/aoni02/katayama/dataset/DATA2019/sp/*csv'))
    feature_files = sorted(glob.glob('/mnt/aoni02/katayama/dataset/DATA2019/feature/*csv'))
    lang_files = sorted(glob.glob('/mnt/aoni02/katayama/dataset/DATA2019/decode_new/*csv'))
    x_spA=x_spB=y1=y2=[]
    x_img=y3=[]
    x_target=y=[]
    X_pre=X=X_id=X_pre_id=[]
    timestep = 10
    sp_step = 1
    for sp_f,f,lf in zip(sp_files[args.input1:args.input2],feature_files[args.input1:args.input2],lang_files[args.input1:args.input2]):
        df = pd.read_csv(sp_f)
        df = df[:len(df)//sp_step*sp_step]
        x_spA = np.append(x_spA,[df.iloc[:,:256].values[i:i+sp_step*timestep] for i in range(0,len(df)-sp_step*timestep,sp_step)],axis=0) if len(x_spA) != 0 else np.array([df.iloc[:,:256].values[i:i+sp_step*timestep] for i in range(0,len(df)-sp_step*timestep,sp_step)])
        x_spB = np.append(x_spB,[df.iloc[:,256:].values[i:i+sp_step*timestep] for i in range(0,len(df)-sp_step*timestep,sp_step)],axis=0) if len(x_spB) != 0 else np.array([df.iloc[:,256:].values[i:i+sp_step*timestep] for i in range(0,len(df)-sp_step*timestep,sp_step)])
        
        length = len(df) // sp_step
        #length = len(df)
        df = pd.read_csv(f).iloc[:length,:]
        y1 = np.append(y1,[df['utter_A'].values[i:i+timestep] for i in range(len(df)-timestep)],axis=0) if len(y1) != 0 else np.array([df['utter_A'].values[i:i+timestep] for i in range(len(df)-timestep)])
        y2 = np.append(y2,[df['utter_B'].values[i:i+timestep] for i in range(len(df)-timestep)],axis=0) if len(y2) != 0 else np.array([df['utter_B'].values[i:i+timestep] for i in range(len(df)-timestep)])
        y3 = np.append(y3,[df['gaze'].values[i:i+timestep] for i in range(len(df)-timestep)],axis=0) if len(y3) != 0 else np.array([df['gaze'].values[i:i+timestep] for i in range(len(df)-timestep)])
        x_img = np.append(x_img,load_image(df['path'].values),axis=0) if len(x_img) != 0 else load_image(df['path'].values)
        label = df['target'].map(lambda x:0 if x =='A' else 1).values
        #x_target = np.append(x_target,label[:length-1],axis=0) if len(x_target) != 0 else label[:length-1]
        x_target = np.append(x_target,[label[i:i+timestep] for i in range(len(label)-timestep)],axis=0) if len(x_target) != 0 else np.array([label[i:i+timestep] for i in range(len(label)-timestep)])
        label = df['action'].map(lambda x:1 if x =='Passive' else 2 if 'Continue' in x else 0).values
        label = add_positive_label(label)
        #y = np.append(y,label[1:length]) if len(y) != 0 else label[1:length]
        y = np.append(y,label[timestep:]) if len(y) != 0 else label[timestep:]
        
        """df = pd.read_csv(lf)[:length]
        X_pre = np.append(X_pre,[df['pre_content'].values[i+timestep-1] for i in range(len(df)-timestep)],axis=0) \
               if len(X_pre) != 0 else np.array([df['pre_content'].values[i+timestep-1] for i in range(len(df)-timestep)])
        X = np.append(X,[df['content'].values[i+timestep-1] for i in range(len(df)-timestep)],axis=0) \
               if len(X) != 0 else np.array([df['content'].values[i+timestep-1] for i in range(len(df)-timestep)])
        X_id = np.append(X_id,[df['ID'].values[i+timestep-1] for i in range(len(df)-timestep)],axis=0) \
               if len(X_id) != 0 else np.array([df['ID'].values[i+timestep-1] for i in range(len(df)-timestep)])
        X_pre_id = np.append(X_pre_id,[df['pre_ID'].values[i+timestep-1] for i in range(len(df)-timestep)],axis=0) \
               if len(X_pre_id) != 0 else np.array([df['pre_ID'].values[i+timestep-1] for i in range(len(df)-timestep)])"""
    #feature = np.array([word2id(w.split()) for w in wakati(X)])
    #feature_pre = np.array([word2id(w.split()) for w in wakati(X_pre)])
    #x_spA = x_spA.reshape(-1,10,2,256)
    #x_spB = x_spB.reshape(-1,10,2,256)
    x_target = x_target.reshape(-1,timestep,1)
    #y = y.reshape(-1,1)
    y1 = y1.reshape(-1,timestep,1)
    y2 = y2.reshape(-1,timestep,1)
    y3 = y3.reshape(-1,timestep,1)
    print(x_spA.shape,x_img.shape,y.shape,x_target.shape,y1.shape,y2.shape,y3.shape)
    
    model = PassiveTrain()
    model.summary()
    print(Counter(y))
    index = np.where(y!=2)[0]
    print(Counter(y[index]))
    print('label_weight:',{0:1.0 , 1: Counter(y[index])[0] /  Counter(y[index])[1]})
    model_save = keras.callbacks.ModelCheckpoint(filepath= args.out + "/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss',save_weights_only=True)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_passive_loss',patience=args.stop_trigger,verbose=1)
    hist = model.fit([x_spA[index],
                      x_spB[index],
                      x_img[index],
                      x_target[index],
                      #feature_pre[index],
                      #feature[index],
                      #X_pre_id[index],
                      #X_id[index]
                     ]
                     ,[y[index],y1[index],y2[index],y3[index]],
          epochs=args.epoch,
          batch_size=args.batchsize,
          callbacks = [
              early_stopping,
              model_save
                              ],
          class_weight = {"passive":{0:1.0 , 1: Counter(y[index])[0] /  Counter(y[index])[1]}},
          validation_split=0.2)
    
if __name__ == '__main__':
    main()