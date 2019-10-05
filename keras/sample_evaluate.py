from keras_model import *
import keras
from keras.layers import *
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="1", # specify GPU number
        allow_growth=True
    )
)
set_session(tf.Session(config=config))

import sys
sys.path.append('../keras/')
#from vad_predict import Utterance,TimeUtterance20
#from gaze_train import TimeGazeTrain
from NLP import *
#from passive_train import PassiveTrain

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
    Error = []
    for i in range(len(y)):
        if y_pred[i] == 0: continue
        if 1 in y[i-windowsize:i+windowsize]:
            correct += 1
            error_dist = np.where(y[i-windowsize:i+windowsize]==1)[0][0] - windowsize
            Error.append(error_dist)
    precision = correct / np.sum(y_pred>0) if y_pred.sum() != 0 else 0
    recall = correct / np.sum(y==1)
    
    #print('precision:',precision,'recall:',recall)
    return precision,recall,Error
    
import glob
import pandas as pd
import cv2
import numpy as np
from collections import Counter
if __name__ == '__main__':
    sp_files = sorted(glob.glob('/mnt/aoni02/katayama/dataset/DATA2019/sp/*csv'))
    feature_files = sorted(glob.glob('/mnt/aoni02/katayama/dataset/DATA2019/feature/*csv'))
    lang_files = sorted(glob.glob('/mnt/aoni02/katayama/dataset/DATA2019/decode_new/*csv'))
    x_spA=x_spB=y1=y2=[]
    x_img=y3=[]
    X=X_pre=X_pre_id=X_id=[]
    x_target=y=[]
    timestep=10
    sp_step = 2
    for sp_f,f,lf in zip(sp_files[83:],feature_files[83:],lang_files[83:]):
        df = pd.read_csv(sp_f)
        df = df[:len(df)//sp_step*sp_step]
        x_spA = np.append(x_spA,[df.iloc[:,:256].values[i:i+sp_step*timestep] for i in range(0,len(df)-sp_step*timestep,sp_step)],axis=0) if len(x_spA) != 0 else np.array([df.iloc[:,:256].values[i:i+sp_step*timestep] for i in range(0,len(df)-sp_step*timestep,sp_step)])
        x_spB = np.append(x_spB,[df.iloc[:,256:].values[i:i+sp_step*timestep] for i in range(0,len(df)-sp_step*timestep,sp_step)],axis=0) if len(x_spB) != 0 else np.array([df.iloc[:,256:].values[i:i+sp_step*timestep] for i in range(0,len(df)-sp_step*timestep,sp_step)])
        
        length = len(df) // sp_step
        df = pd.read_csv(f).iloc[:length,:]
        
        #y1 = np.append(y1,df['utter_A'].values[:length-1]) if len(y1) != 0 else df['utter_A'].values[:length-1]
        #y2 = np.append(y2,df['utter_B'].values[:length-1]) if len(y2) != 0 else df['utter_B'].values[:length-1]
        #y3 = np.append(y3,df['gaze'].values[:length-1]) if len(y3) != 0 else df['gaze'].values[:length-1]
        y1 = np.append(y1,[df['utter_A'].values[i:i+timestep] for i in range(len(df)-timestep)],axis=0) if len(y1) != 0 else np.array([df['utter_A'].values[i:i+timestep] for i in range(len(df)-timestep)])
        y2 = np.append(y2,[df['utter_B'].values[i:i+timestep] for i in range(len(df)-timestep)],axis=0) if len(y2) != 0 else np.array([df['utter_B'].values[i:i+timestep] for i in range(len(df)-timestep)])
        y3 = np.append(y3,[df['gaze'].values[i:i+timestep] for i in range(len(df)-timestep)],axis=0) if len(y3) != 0 else np.array([df['gaze'].values[i:i+timestep] for i in range(len(df)-timestep)])
        x_img = np.append(x_img,load_image(df['path'].values),axis=0) if len(x_img) != 0 else load_image(df['path'].values)
        label = df['target'].map(lambda x:0 if x =='A' else 1).values
        
        x_target = np.append(x_target,[label[i:i+timestep] for i in range(len(label)-timestep)],axis=0) if len(x_target) != 0 else np.array([label[i:i+timestep] for i in range(len(label)-timestep)])
        label = df['action'].map(lambda x:1 if x =='Passive' else 2 if 'Continue' in x else 0).values
        #label = add_positive_label(label)
        
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
    x_spA = x_spA.reshape(-1,10,2,256)
    x_spB = x_spB.reshape(-1,10,2,256)
    x_target = x_target.reshape(-1,timestep,1)
    #y = y.reshape(-1,1)
    y1 = y1.reshape(-1,timestep,1)
    y2 = y2.reshape(-1,timestep,1)
    y3 = y3.reshape(-1,timestep,1)
    index = np.where(y!=2)[0]
    print(Counter(y[index]))
    model = PassiveTrain()
    #model = ActiveTrain()
    model.summary()
    Recall = []
    Precision = []
    F1 = []
    Error = []
    for num in [5,10,15,20]:#,25,30,35,40]:
      print('permission window size is',num)
      max_pre = 0
      max_rec = 0
      max_f1 = 0
      PATH_list = sorted(glob.glob('result/passive_sp20/train13-83/weights*'))
      for PATH in PATH_list:
          model.load_weights(PATH)
          y_pred1 = model.predict([x_spA[index],x_spB[index],x_img[index],x_target[index]])
          #y_pred1 = model.predict([x_spA[index],x_spB[index],x_img[index],x_target[index], \
          #                                     feature_pre[index],feature[index],X_pre_id[index],X_id[index]
          #                                    ])
          y_aft1 = filetering(np.reshape(y_pred1[0],(-1,)))

          #print(PATH)
          precision,recall,error = report_recall_and_precision(y[index],y_aft1,windowsize=num)
          #print()
          f1 = 2 * precision * recall / (precision + recall + 1e-08) 
          
          if max_f1 < f1:
            max_pre = precision
            max_rec = recall
            max_f1 = f1
            max_error = error
           
      Recall.append(max_rec)
      Precision.append(max_pre)
      F1.append(max_f1)
      Error.append(','.join(list(map(str,max_error))))
    df = pd.DataFrame({'precision':Precision,'recall':Recall,'f1':F1,'error':Error},)
    index = [5,10,15,20]#,25,30,35,40]
    df['window'] = index
    df.to_csv('./evaluate/passive_lang_not_freeze_sp20_result2.csv',index=False)
