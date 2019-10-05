import keras
from keras.layers import *
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="0", # specify GPU number
        allow_growth=True
    )
)
set_session(tf.Session(config=config))

from vad_predict import Utterance
from gaze_train import TimeGazeTrain

def ActiveTrain(timestep=10):
    input_spA = Input(shape=(timestep,256))
    input_spB = Input(shape=(timestep,256))
    utterance1 = Utterance(input_spA)
    utterance1.load_weights('result/utterance/weights.10-0.47-0.79.h5')
    x_spA = utterance1.layers[-3].output
    utterance2 = Utterance(input_spB)
    utterance2.load_weights('result/utterance/weights.10-0.47-0.79.h5')
    x_spB = utterance2.layers[-3].output
    y_spA = Dense(1,activation='sigmoid',name='spA')(x_spA)
    y_spB = Dense(1,activation='sigmoid',name='spB')(x_spB)
    input_img = Input(shape=(timestep,32,96,1))
    gaze = TimeGazeTrain(input_img)
    gaze.load_weights('result/gaze/weights.03-0.16-0.93.h5')
    img = gaze.layers[-3].output
    y_gaze = Dense(1,activation='sigmoid',name='gaze')(img)
    input_target = Input(shape=(timestep,1))
    
    x = concatenate([x_spA,x_spB,img,input_target])
    #x = Dense(256,activation='relu')(x)
    #x = BatchNormalization()(x)
    x = Dense(64,activation='relu',
                  kernel_regularizer=regularizers.l2(0.01),
                  #activity_regularizer=regularizers.l2(0.01),
                  kernel_initializer='random_uniform',
                   )(x)
    #x = BatchNormalization()(x)
    x = LSTM(64,activation='tanh',recurrent_dropout=0.5,dropout=0.5,
            kernel_regularizer=regularizers.l2(0.01),
            kernel_initializer='random_uniform', 
            )(x)
    x = BatchNormalization()(x)
    x = Dense(64,activation='relu',
                  kernel_regularizer=regularizers.l2(0.01),
                  #activity_regularizer=regularizers.l2(0.01),
                  kernel_initializer='random_uniform',
                   )(x)
    #x = BatchNormalization()(x)
    y = Dense(1,activation='sigmoid',name='active')(x)
    model = keras.Model(inputs=[input_spA,
                                                input_spB,
                                                input_img,
                                                input_target],
                                                outputs=[y,y_spA,y_spB,y_gaze]
                                                )
    #for layer in model.layers[:21]:
    #    layer.trainable = False
    model.compile(loss='binary_crossentropy',
                  optimizer = keras.optimizers.Adam(1e-04),
                  metrics=['accuracy'],
                  loss_weights = {'active':1.0,'spA':0.25,'spB':0.25,"gaze":0.5}
                 )
    
    return model

def load_image(paths,gray_flag=0,timestep=10):
    #pathを受け取って画像を返す
    img_feature = []
    for path in paths:
        x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if x is None:
            x = np.array([0]*32*96)
        x = x.reshape(32,96,1)
        img_feature.append(x / 255.0)
    return np.array([img_feature[i:i+timestep] for i in range(len(paths)-timestep)],dtype=np.float32)

def add_positive_label(y):
    y_after = [0]*len(y)
    for i in range(len(y)):
        if y[i] != 1: continue
        for j in range(max(i-5,0),min(len(y),i+5)):
            y_after[j] = 1
    return y_after

import glob
import pandas as pd
import cv2
import numpy as np
from collections import Counter
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each omini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=50,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result/active/train13-60/',
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
    sp_files = sorted(glob.glob('/mnt/aoni02/katayama/dataset/DATA2019/sp/*csv'))
    feature_files = sorted(glob.glob('/mnt/aoni02/katayama/dataset/DATA2019/feature/*csv'))

    x_spA=x_spB=y1=y2=[]
    x_img=y3=[]
    x_target=y=[]
    timestep = 10
    for sp_f,f in zip(sp_files[args.input1:args.input2],feature_files[args.input1:args.input2]):
        df = pd.read_csv(sp_f)
        x_spA = np.append(x_spA,[df.iloc[:,:256].values[i:i+timestep] for i in range(len(df)-timestep)],axis=0) if len(x_spA) != 0 else np.array([df.iloc[:-1,:256].values[i:i+timestep] for i in range(len(df)-timestep)])
        x_spB = np.append(x_spB,[df.iloc[:,256:].values[i:i+timestep] for i in range(len(df)-timestep)],axis=0) if len(x_spB) != 0 else np.array([df.iloc[:-1,256:].values[i:i+timestep] for i in range(len(df)-timestep)])
        #x_spA = np.append(x_spA,df.iloc[:-1,:256].values,axis=0) if len(x_spA) != 0 else df.iloc[:,:256].values[:-1]
        #x_spB = np.append(x_spB,df.iloc[:-1,256:].values,axis=0) if len(x_spB) != 0 else df.iloc[:,256:].values[:-1]
        length = len(df)
        df = pd.read_csv(f).iloc[:length,:]
        #print(len(df),length)
        #y1 = np.append(y1,df['utter_A'].values[:length-1]) if len(y1) != 0 else df['utter_A'].values[:length-1]
        #y2 = np.append(y2,df['utter_B'].values[:length-1]) if len(y2) != 0 else df['utter_B'].values[:length-1]
        #y3 = np.append(y3,df['gaze'].values[:length-1]) if len(y3) != 0 else df['gaze'].values[:length-1]
        y1 = np.append(y1,[df['utter_A'].values[i:i+timestep] for i in range(len(df)-timestep)],axis=0) if len(y1) != 0 else np.array([df['utter_A'].values[i:i+timestep] for i in range(len(df)-timestep)])
        y2 = np.append(y2,[df['utter_B'].values[i:i+timestep] for i in range(len(df)-timestep)],axis=0) if len(y2) != 0 else np.array([df['utter_B'].values[i:i+timestep] for i in range(len(df)-timestep)])
        y3 = np.append(y3,[df['gaze'].values[i:i+timestep] for i in range(len(df)-timestep)],axis=0) if len(y3) != 0 else np.array([df['gaze'].values[i:i+timestep] for i in range(len(df)-timestep)])
        x_img = np.append(x_img,load_image(df['path'].values),axis=0) if len(x_img) != 0 else load_image(df['path'].values)
        label = df['target'].map(lambda x:0 if x =='A' else 1).values
        #x_target = np.append(x_target,label[:length-1],axis=0) if len(x_target) != 0 else label[:length-1]
        x_target = np.append(x_target,[label[i:i+timestep] for i in range(len(label)-timestep)],axis=0) if len(x_target) != 0 else np.array([label[i:i+timestep] for i in range(len(label)-timestep)])
        label = df['action'].map(lambda x:1 if 'Active' in x else 2 if 'Continue' in x else 0).values
        print(Counter(label))
        #label = add_positive_label(label)
        #y = np.append(y,label[1:length]) if len(y) != 0 else label[1:length]
        y = np.append(y,label[timestep:]) if len(y) != 0 else label[timestep:]
    
    x_target = x_target.reshape(-1,timestep,1)
    #y = y.reshape(-1,1)
    y1 = y1.reshape(-1,timestep,1)
    y2 = y2.reshape(-1,timestep,1)
    y3 = y3.reshape(-1,timestep,1)
    print(x_spA.shape,x_img.shape,y.shape,x_target.shape,y1.shape,y2.shape,y3.shape)
    train_index = np.where(y!=2)[0]
    np.random.shuffle(train_index)
    model = ActiveTrain()
    model.summary()
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,verbose=1)
    model_save = keras.callbacks.ModelCheckpoint(filepath= args.out + "/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss',save_weights_only=True)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=args.stop_trigger,verbose=1)
    hist = model.fit([x_spA[train_index],
                      x_spB[train_index],
                      x_img[train_index],
                      x_target[train_index]
                     ],[y[train_index],y1[train_index],y2[train_index],y3[train_index]],
          epochs=args.epoch,
          batch_size=args.batchsize,
          callbacks = [
              early_stopping,
              model_save
                              ],
          validation_split=0.33)
    #model.save('passive.h5')
if __name__ == '__main__':
    main()