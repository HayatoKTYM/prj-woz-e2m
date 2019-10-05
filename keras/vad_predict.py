__author__ = 'Hayato Katayama'
from keras_model import *
import keras

from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
import glob
import pandas as pd
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each omini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=150,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result/utterance',
                        help='Directory to output the result')
    parser.add_argument('--resume_model', type=str)
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--stop_trigger', type=int, default=15)
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')
    
    sp_files = sorted(glob.glob('/mnt/aoni02/katayama/dataset/DATA2019/sp/*csv'))
    feature_files = sorted(glob.glob('/mnt/aoni02/katayama/dataset/DATA2019/feature/*csv'))
    sp_step = 1
    x=y=Acc=[]
    assert len(sp_files) == len(feature_files), print('file path is not correct.')
    for sp_f,f in zip(sp_files[13:],feature_files[13:]):
        df = pd.read_csv(sp_f)
        
        df = df[:len(df)//sp_step*sp_step]
        x = np.append(x,df.iloc[:,:256].values,axis=0) if len(x) != 0 else df.iloc[:,:256].values 
        x = np.append(x,df.iloc[:,256:].values,axis=0) if len(x) != 0 else df.iloc[:,256:].values
        
        length = len(df) // sp_step
        #length = len(df)
        df = pd.read_csv(f)
        #print(length,len(df))
        assert len(df) >= length, print(sp_f)
        y = np.append(y,df['utter_A'].values[1:length+1]) if len(y) != 0 else df['utter_A'].values[1:length+1]
        y = np.append(y,df['utter_B'].values[1:length+1]) if len(y) != 0 else df['utter_B'].values[1:length+1]
        x = x[:sp_step*len(y)]
    #x = sc.fit_transform(x)
    #x = x.reshape(-1,sp_step,256)
    print(x.shape,y.shape)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=args.stop_trigger,verbose=1)
    model_save = keras.callbacks.ModelCheckpoint(filepath=args.out+"/weights.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.h5", monitor='val_loss',save_weights_only=True)
    
    #交差検定
    for i in range(1):
        x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.2,random_state=i*2+1)
        model = Utterance()
        model.summary()
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=args.stop_trigger,verbose=1)
        hist = model.fit(x_train,y_train,
          epochs=args.epoch,
          batch_size=args.batchsize,
          callbacks = [
              early_stopping,
              model_save
                      ],
          validation_split=0.25)
        acc = model.evaluate(x_val,y_val)[1]
        Acc.append(acc)
        print(np.average(acc))
        
if __name__ == '__main__':
    main()
    

