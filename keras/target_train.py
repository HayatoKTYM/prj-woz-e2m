from keras_model import *
import keras

def TargetTrain():
    input_spA = Input(shape=(256,))
    input_spB = Input(shape=(256,))
    utterance1 = Utterance(input_spA)
    utterance1.load_weights('result/utterance/weights.57-0.42-0.81.h5')
    x_spA = utterance1.layers[-2].output
    utterance2 = Utterance(input_spB)
    utterance2.load_weights('result/utterance/weights.57-0.42-0.81.h5')
    x_spB = utterance2.layers[-2].output
    y_spA = Dense(1,activation='sigmoid',name='spA')(x_spA)
    y_spB = Dense(1,activation='sigmoid',name='spB')(x_spB)
    input_img = Input(shape=(32,96,1))
    gaze = GazeTrain(input_img)
    gaze.load_weights('result/gaze/weights.03-0.16-0.93.h5')
    img = gaze.layers[-2].output
    y_gaze = Dense(1,activation='sigmoid',name='gaze')(img)
    input_target = Input(shape=(1,))
    
    x = concatenate([x_spA,x_spB,img,input_target])
    x = Dense(64,activation='relu',
                  kernel_regularizer=regularizers.l2(0.01),
                  kernel_initializer='random_uniform',
                   )(x)
    x = BatchNormalization()(x)
    x = Dense(64,activation='relu',
                  kernel_regularizer=regularizers.l2(0.01),
                  kernel_initializer='random_uniform',
                   )(x)
    x = BatchNormalization()(x)
    y = Dense(1,activation='sigmoid',name='target')(x)
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
                  loss_weights = {'target':1.0,'spA':0.25,'spB':0.25,"gaze":0.25}
                 )
    
    return model

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

import glob
import pandas as pd
import cv2
import numpy as np
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each omini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=50,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result/target/train13-60/',
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

    x_spA=x_spB=y1=y2=[]
    x_img=y3=[]
    x_target=y=[]

    for sp_f,f in zip(sp_files[args.input1:args.input2],feature_files[args.input1:args.input2]):
        df = pd.read_csv(sp_f)
        x_spA = np.append(x_spA,df.iloc[:-1,:256].values,axis=0) if len(x_spA) != 0 else df.iloc[:,:256].values[:-1]
        x_spB = np.append(x_spB,df.iloc[:-1,256:].values,axis=0) if len(x_spB) != 0 else df.iloc[:,256:].values[:-1]
        length = len(df)

        df = pd.read_csv(f)
        y1 = np.append(y1,df['utter_A'].values[:length-1]) if len(y1) != 0 else df['utter_A'].values[:length-1]
        y2 = np.append(y2,df['utter_B'].values[:length-1]) if len(y2) != 0 else df['utter_B'].values[:length-1]
        y3 = np.append(y3,df['gaze'].values[:length-1]) if len(y3) != 0 else df['gaze'].values[:length-1]
        x_img = np.append(x_img,load_image(df['path'].values[:length-1]),axis=0) if len(x_img) != 0 else load_image(df['path'].values[:length-1])
        label = df['target'].map(lambda x:0 if x =='A' else 1).values
        x_target = np.append(x_target,label[:length-1],axis=0) if len(x_target) != 0 else label[:length-1]
        y = np.append(y,label[1:length]) if len(y) != 0 else label[1:length]
    print(x_spA.shape,x_img.shape,y.shape)
    x_target = x_target.reshape(-1,1)
    model = TargetTrain()
    model.summary()
    model_save = keras.callbacks.ModelCheckpoint(filepath= args.out + "/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss',save_weights_only=True)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=args.stop_trigger,verbose=1)
    hist = model.fit([x_spA,x_spB,x_img,x_target],[y,y1,y2,y3],
          epochs=args.epoch,
          batch_size=args.batchsize,
          callbacks = [
              early_stopping,
              model_save],
          validation_split=0.33)
    
if __name__ == '__main__':
    main()