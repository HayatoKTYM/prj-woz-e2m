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

import sys
#sys.path.append('../keras/')
from vad_predict import Utterance
from gaze_train import TimeGazeTrain
from NLP import *

def PassiveTrain(timestep=10):
    input_spA = Input(shape=(timestep,256))
    input_spB = Input(shape=(timestep,256))
    utterance1 = Utterance(input_spA)
    #utterance1.load_weights('result/utterance/weights.10-0.47-0.79.h5')
    x_spA = utterance1.layers[-3].output
    utterance2 = Utterance(input_spB)
    #utterance2.load_weights('result/utterance/weights.10-0.47-0.79.h5')
    x_spB = utterance2.layers[-3].output
    y_spA = Dense(1,activation='sigmoid',name='spA')(x_spA)
    y_spB = Dense(1,activation='sigmoid',name='spB')(x_spB)
    input_img = Input(shape=(timestep,32,96,1))
    gaze = TimeGazeTrain(input_img)
    #gaze.load_weights('result/gaze/weights.03-0.16-0.93.h5')
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
    y = Dense(1,activation='sigmoid',name='passive')(x)
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
                  loss_weights = {'passive':1.0,'spA':0.25,'spB':0.25,"gaze":0.5}
                 )
    
    return model

def PassiveTrain(timestep=10):
    input_spA = Input(shape=(timestep,256))
    input_spB = Input(shape=(timestep,256))
    utterance1 = Utterance(input_spA,freeze=True)
    utterance1.load_weights('result/utterance/weights.10-0.47-0.79.h5')
    x_spA = utterance1.layers[-3].output
    utterance2 = Utterance(input_spB,freeze=True)
    utterance2.load_weights('result/utterance/weights.10-0.47-0.79.h5')
    x_spB = utterance2.layers[-3].output
    y_spA = Dense(1,activation='sigmoid',name='spA')(x_spA)
    y_spB = Dense(1,activation='sigmoid',name='spB')(x_spB)
    input_img = Input(shape=(timestep,32,96,1))
    gaze = TimeGazeTrain(input_img,freeze=True)
    gaze.load_weights('result/gaze/weights.03-0.16-0.93.h5')
    img = gaze.layers[-3].output
    y_gaze = Dense(1,activation='sigmoid',name='gaze')(img)
    input_target = Input(shape=(timestep,1))
    input_lang1 = Input(shape=(10,),name='pre_content')
    input_lang2 = Input(shape=(10,),name='content')
    input_lang3 = Input(shape=(1,),name='id1')
    input_lang4 = Input(shape=(1,),name='id2')
    move = moveModel(input_lang1,input_lang2,input_lang3,input_lang4)
    move = move.layers[-2].output
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
    x = concatenate([x,move])
    x = Dense(64,activation='relu',
                  kernel_regularizer=regularizers.l2(0.01),
                  #activity_regularizer=regularizers.l2(0.01),
                  kernel_initializer='random_uniform',
                   )(x)
    #x = BatchNormalization()(x)
    y = Dense(1,activation='sigmoid',name='passive')(x)
    model = keras.Model(inputs=[input_spA,
                                                input_spB,
                                                input_img,
                                                input_target,
                                                input_lang1,
                                                input_lang2,
                                                input_lang3,
                                                input_lang4],
                                                outputs=[y,y_spA,y_spB,y_gaze]
                                                )
    #for layer in model.layers[:21]:
    #    layer.trainable = False
    model.compile(loss='binary_crossentropy',
                  optimizer = keras.optimizers.Adam(1e-04),
                  metrics=['accuracy'],
                  loss_weights = {'passive':1.0,'spA':0.25,'spB':0.25,"gaze":0.5}
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

def filetering(y,N=30):
    """
    行動生成したら，N秒間行動しない処理
    """
    y_aft = np.copy(y)
    for i in range(len(y)):
        if y[i] < 0.5: 
            y_aft[i] = 0
        else:
            for j in range(i+1,min(i+N,len(y))):
                y_aft[j] = 0
    return y_aft

def report_recall_and_precision(y,y_pred,windowsize=5):
    correct = 0
    negative_correct = 0
    for i in range(len(y)):
        if y_pred[i] == 0: continue
        if 1 in y[i-windowsize:i+windowsize]:
            correct += 1
        
    
    precision = correct / np.sum(y_pred>0)
    recall = correct / np.sum(y==1)
    
    print('precision:',precision,'recall:',recall)
    
import glob
import pandas as pd
import cv2
import numpy as np
if __name__ == '__main__':
    sp_files = sorted(glob.glob('/mnt/aoni02/katayama/dataset/DATA2019/sp/*csv'))
    feature_files = sorted(glob.glob('/mnt/aoni02/katayama/dataset/DATA2019/feature/*csv'))
    lang_files = sorted(glob.glob('/mnt/aoni02/katayama/dataset/DATA2019/decode_new/*csv'))
    x_spA=x_spB=y1=y2=[]
    x_img=y3=[]
    X=X_pre=X_pre_id=X_id=[]
    x_target=y=[]
    timestep=10
    for sp_f,f,lf in zip(sp_files[80:],feature_files[80:],lang_files[80:]):
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
        label = df['action'].map(lambda x:1 if x =='Passive' else 0).values
        #label = add_positive_label(label)
        #y = np.append(y,label[1:length]) if len(y) != 0 else label[1:length]
        y = np.append(y,label[timestep:]) if len(y) != 0 else label[timestep:]
        df = pd.read_csv(lf)[:length]
        X_pre = np.append(X_pre,[df['pre_content'].values[i+timestep-1] for i in range(len(df)-timestep)],axis=0) \
                if len(X_pre) != 0 else np.array([df['pre_content'].values[i+timestep-1] for i in range(len(df)-timestep)])
        X = np.append(X,[df['content'].values[i+timestep-1] for i in range(len(df)-timestep)],axis=0) \
                if len(X) != 0 else np.array([df['content'].values[i+timestep-1] for i in range(len(df)-timestep)])
        X_id = np.append(X_id,[df['ID'].values[i+timestep-1] for i in range(len(df)-timestep)],axis=0) \
                if len(X_id) != 0 else np.array([df['ID'].values[i+timestep-1] for i in range(len(df)-timestep)])
        X_pre_id = np.append(X_pre_id,[df['pre_ID'].values[i+timestep-1] for i in range(len(df)-timestep)],axis=0) \
                if len(X_pre_id) != 0 else np.array([df['pre_ID'].values[i+timestep-1] for i in range(len(df)-timestep)])
    feature = np.array([word2id(w.split()) for w in wakati(X)])
    feature_pre = np.array([word2id(w.split()) for w in wakati(X_pre)])
    x_target = x_target.reshape(-1,timestep,1)
    #y = y.reshape(-1,1)
    y1 = y1.reshape(-1,timestep,1)
    y2 = y2.reshape(-1,timestep,1)
    y3 = y3.reshape(-1,timestep,1)

    model = PassiveTrain()
    #model = ActiveTrain()
    model.summary()
    
    PATH_list = sorted(glob.glob('result/passive_lang/train0-80/weights*'))
    for PATH in PATH_list:
        model.load_weights(PATH)
        y_pred1 = model.predict([x_spA,x_spB,x_img,x_target,feature_pre,feature,X_pre_id,X_id])
        y_aft1 = filetering(np.reshape(y_pred1[0],(-1,)))

        print(PATH)
        report_recall_and_precision(y,y_aft1,windowsize=30)
        print()