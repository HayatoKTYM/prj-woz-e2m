import keras
from keras.layers import *
import keras.backend as K
from keras_self_attention import SeqSelfAttention
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="0", # specify GPU number
        allow_growth=True
    )
)
set_session(tf.Session(config=config))


def Utterance(input_spA = Input(shape=(256,)),freeze=False):
    x = Dense(256,activation='relu')(input_spA)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256,activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(64,activation='relu')(x)
    x = BatchNormalization()(x)
    out = Dense(1,activation='sigmoid')(x)
    model = keras.Model(input_spA,out)
    model.compile(loss='binary_crossentropy',
                  optimizer = keras.optimizers.Adam(1e-03),
                  metrics=['accuracy'])
    if freeze:
        for layer in model.layers[:-4]:
            layer.trainable = False
    return model

def Utterance20(input_spA = Input(shape=(2,256,)),freeze=False):
    x = Dense(512,activation='relu')(input_spA)
    #x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256,activation='relu')(x)
    #x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    #x = Dense(32,activation='relu')(x)
    #x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)
    x = LSTM(256,dropout=0.5,recurrent_dropout=0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(64,activation='relu')(x)
    x = BatchNormalization()(x)
    out = Dense(1,activation='sigmoid')(x)
    model = keras.Model(input_spA,out)
    model.compile(loss='binary_crossentropy',
                  optimizer = keras.optimizers.Adam(1e-03),
                  #optimizer = keras.optimizers.SGD(1e-02),
                  metrics=['accuracy'])
    if freeze:
        for layer in model.layers[:4]:
            layer.trainable = False
    return model

def Utterance20_dense(input_spA = Input(shape=(2,256,)),freeze=False):
    x = Dense(256,activation='relu')(input_spA)
    x = Dropout(0.5)(x)
    x = Dense(128,activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Reshape((-1,))(x)
    x = Dense(128,activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64,activation='relu')(x)
    x = BatchNormalization()(x)
    out = Dense(1,activation='sigmoid')(x)
    model = keras.Model(input_spA,out)
    model.compile(loss='binary_crossentropy',
                  optimizer = keras.optimizers.Adam(1e-03),
                  #optimizer = keras.optimizers.SGD(1e-02),
                  metrics=['accuracy'])
    if freeze:
        for layer in model.layers[:4]:
            layer.trainable = False
    return model

def Utterance100(input_spA = Input(shape=(10,256,)),freeze=False):
    x = Dense(512,activation='relu')(input_spA)
    #x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256,activation='relu')(x)
    #x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    #x = Dense(32,activation='relu')(x)
    #x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)
    x = LSTM(256,dropout=0.5,recurrent_dropout=0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(64,activation='relu')(x)
    x = BatchNormalization()(x)
    out = Dense(1,activation='sigmoid')(x)
    model = keras.Model(input_spA,out)
    model.compile(loss='binary_crossentropy',
                  optimizer = keras.optimizers.Adam(1e-03),
                  #optimizer = keras.optimizers.SGD(1e-02),
                  metrics=['accuracy'])
    if freeze:
        for layer in model.layers[:4]:
            layer.trainable = False
    return model

def TimeUtterance(input_spA = Input(shape=(256,)),trainable=True):
    x = TimeDistributed(Dense(256,activation='relu',trainable=trainable),trainable=trainable)(input_spA)
    x = TimeDistributed(BatchNormalization(trainable=trainable),trainable=trainable)(x)
    x = TimeDistributed(Dense(256,activation='relu',trainable=trainable),trainable=trainable)(x)
    x = TimeDistributed(BatchNormalization(trainable=trainable),trainable=trainable)(x)
    x = TimeDistributed(Dense(64,activation='relu',trainable=trainable),trainable=True)(x)
    x = TimeDistributed(BatchNormalization(trainable=trainable),trainable=trainable)(x) 
    out = TimeDistributed(Dense(1,activation='sigmoid'))(x)
    model = keras.Model(input_spA,out)
    
    return model

def TimeUtterance20(input_spA = Input(shape=(2,256,)),trainable=True):
    x = TimeDistributed(Dense(512,activation='relu',trainable=trainable),trainable=trainable)(input_spA)
    #x = BatchNormalization()(x)
    x = TimeDistributed(Dropout(0.5))(x)
    x = TimeDistributed(Dense(256,activation='relu',trainable=trainable),trainable=trainable)(x)
    #x = BatchNormalization()(x)
    x = TimeDistributed(Dropout(0.5))(x)
    #x = Dense(32,activation='relu')(x)
    #x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)
    x = TimeDistributed(LSTM(256,dropout=0.5,recurrent_dropout=0.5,trainable=trainable),trainable=trainable)(x)
    x = TimeDistributed(BatchNormalization(trainable=trainable),trainable=trainable)(x)
    x = TimeDistributed(Dense(32,activation='relu',trainable=trainable),trainable=True)(x)
    x = TimeDistributed(BatchNormalization(trainable=trainable),trainable=True)(x)
    out = TimeDistributed(Dense(1,activation='sigmoid'))(x)
    model = keras.Model(input_spA,out)
    #model.summary()
    #model.compile(loss='binary_crossentropy',
    #              optimizer = keras.optimizers.Adam(1e-03),
    #              #optimizer = keras.optimizers.SGD(1e-02),
    #              metrics=['accuracy'])
    #     if freeze:
    #         for layer in model.layers[:-2]:
    #             layer.trainable = False
    return model

def GazeTrain( input_img = Input(shape=(32,96,1)),freeze=False):
    x = Convolution2D(32,5,activation='relu')(input_img)
    x = Convolution2D(32,5,activation='relu')(x)
    x = MaxPooling2D(2)(x)
    x = BatchNormalization()(x)
    x = Convolution2D(32,5,activation='relu')(x)
    x = Convolution2D(32,5,activation='relu')(x)
    x = MaxPooling2D(2)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(256,activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(32,activation='relu')(x)
    x = Dropout(0.5)(x)
    y = Dense(1,activation='sigmoid')(x)
    model = keras.Model(inputs= input_img,outputs=y)
    model.compile(loss='binary_crossentropy',
                  optimizer = keras.optimizers.Adam(1e-04),
                  metrics=['accuracy'])
    if freeze:
        for layer in model.layers[:-5]:
            layer.trainable = False
    return model

def TimeGazeTrain( input_img = Input(shape=(10,32,96,1)),trainable=True):
    x = TimeDistributed(Convolution2D(32,5,activation='relu',trainable=trainable),trainable=trainable)(input_img)
    x = TimeDistributed(Convolution2D(32,5,activation='relu',trainable=trainable),trainable=trainable)(x)
    x = TimeDistributed(MaxPooling2D(2))(x)
    x = TimeDistributed(BatchNormalization(trainable=trainable),trainable=trainable)(x)
    x = TimeDistributed(Convolution2D(32,5,activation='relu',trainable=trainable),trainable=trainable)(x)
    x = TimeDistributed(Convolution2D(32,5,activation='relu',trainable=trainable),trainable=trainable)(x)
    x = TimeDistributed(MaxPooling2D(2))(x)
    x = TimeDistributed(BatchNormalization(),trainable=trainable)(x)
    x = TimeDistributed(Flatten())(x)
    x = TimeDistributed(Dense(256,activation='relu',trainable=trainable),trainable=trainable)(x)
    x = Dropout(0.5)(x)
    x = TimeDistributed(Dense(32,activation='relu'),trainable=True)(x)
    x = Dropout(0.5)(x)
    y = Dense(1,activation='sigmoid')(x)
    model = keras.Model(inputs= input_img,outputs=y)
    #model.compile(loss='binary_crossentropy',
    #              optimizer = keras.optimizers.Adam(1e-04),
    #              metrics=['accuracy'])
    #if freeze:
    #    for layer in model.layers[:-2]:
    #        layer.trainable = False
    return model

def FaceTrain( input_img = Input(shape=(96,96,1)),freeze=False):
    x = Convolution2D(32,5,activation='relu',
                      #padding='same'
                     )(input_img)
    x = BatchNormalization()(x)
    x = Convolution2D(32,5,activation='relu',
                      #padding='same'
                     )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2)(x)
    x = Convolution2D(32,5,activation='relu',
                      #padding='same'
                     )(x)
    x = BatchNormalization()(x)
    x = Convolution2D(64,5,activation='relu',
                      #padding='same'
                     )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2)(x)
    x = BatchNormalization()(x)
    x = Convolution2D(128,5,activation='relu',
                      #padding='same'
                     )(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(1024,activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256,activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128,activation='relu')(x)
    x = Dropout(0.5)(x)
    y = Dense(1,activation='sigmoid')(x)
    model = keras.Model(inputs= input_img,outputs=y)
    model.compile(loss='binary_crossentropy',
                  optimizer = keras.optimizers.Adam(1e-03),
                  #optimizer = keras.optimizers.SGD(1e-02),
                  metrics=['accuracy'])
    if freeze:
        for layer in model.layers[:-5]:
            layer.trainable = False
    return model

def TimeFaceTrain( input_img = Input(shape=(96,96,1)),trainable=True):
    x = TimeDistributed(Convolution2D(32,5,activation='relu',trainable=trainable
                      #padding='same'
                     ),trainable=trainable)(input_img)
    x = TimeDistributed(BatchNormalization(trainable=trainable),trainable=trainable)(x)
    x = TimeDistributed(Convolution2D(32,5,activation='relu',trainable=trainable
                      #padding='same'
                     ),trainable=trainable)(x)
    x = TimeDistributed(BatchNormalization(trainable=trainable),trainable=trainable)(x)
    x = TimeDistributed(MaxPooling2D(2))(x)
    x = TimeDistributed(Convolution2D(32,5,activation='relu',trainable=trainable
                      #padding='same'
                     ),trainable=trainable)(x)
    x = TimeDistributed(BatchNormalization(trainable=trainable),trainable=trainable)(x)
    x = TimeDistributed(Convolution2D(64,5,activation='relu',trainable=trainable
                      #padding='same'
                     ),trainable=trainable)(x)
    x = TimeDistributed(BatchNormalization(trainable=trainable),trainable=trainable)(x)
    x = TimeDistributed(MaxPooling2D(2))(x)
    x = TimeDistributed(BatchNormalization(trainable=trainable),trainable=trainable)(x)
    x = TimeDistributed(Convolution2D(128,5,activation='relu',trainable=trainable
                      #padding='same'
                     ),trainable=trainable)(x)
    x = TimeDistributed(BatchNormalization(trainable=trainable),trainable=trainable)(x)
    x = TimeDistributed(Flatten())(x)
    x = TimeDistributed(Dense(1024,activation='relu',trainable=trainable),trainable=trainable)(x)
    x = TimeDistributed(Dropout(0.5))(x)
    x = TimeDistributed(Dense(256,activation='relu',trainable=trainable),trainable=trainable)(x)
    x = TimeDistributed(Dropout(0.5))(x)
    x = TimeDistributed(Dense(128,activation='relu'))(x)
    x = TimeDistributed(Dropout(0.5))(x)
    y = TimeDistributed(Dense(1,activation='sigmoid'))(x)
    model = keras.Model(inputs= input_img,outputs=y)
    #     model.compile(loss='binary_crossentropy',
    #                   optimizer = keras.optimizers.Adam(1e-03),
    #                   #optimizer = keras.optimizers.SGD(1e-02),
    #                   metrics=['accuracy'])
    return model

def moveModel(i1 = Input(shape=(10,)),i2 = Input(shape=(10,)),u1_id = Input(shape=(1,)),u2_id = Input(shape=(1,)),trainable=True):
    
    import pickle
    with open('/mnt/aoni02/katayama/new_embedding_metrix.pkl',"rb") as f:
        embedding_metrix = pickle.load(f)
    Embed = keras.layers.Embedding((embedding_metrix.shape[0]), 300,mask_zero = True,weights = [embedding_metrix],input_length=10,trainable=False)
    x_a = Embed(i1)
    x_b = Embed(i2)
    
    shareLSTM = LSTM(64,return_sequences=False,dropout=0.5,recurrent_dropout=0.5,name='lstm',trainable=trainable)
    shareSelfAtt = SeqSelfAttention(units=32,name='self-att',trainable=trainable)
    shareSum = Lambda(lambda a: K.sum(a,axis=-2), output_shape=(64,))
    shareDense = Dense(16,activation='relu',name='dense',trainable=True)
    shareDropout = Dropout(0.5)
    for f in [shareSelfAtt,shareLSTM,shareDense,shareDropout]:
        x_a = f(x_a)
        x_b = f(x_b)
    x = concatenate([x_a,x_b,u1_id,u2_id],axis=-1)
    o = Dense(5,activation='softmax')(x)
    m = keras.Model([i1,i2,u1_id,u2_id],o)
    m.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'],optimizer=keras.optimizers.Adam(1e-04))
    m.load_weights('../keras/move0910.h5',by_name=True)
    
    return m


def PassiveTrain(timestep=10,singletask=False):
    input_spA = Input(shape=(timestep,256))
    input_spB = Input(shape=(timestep,256))
    utterance1 = TimeUtterance(input_spA,trainable=False)
    utterance1.load_weights('result/utterance/weights.32-0.42-0.80.h5')
    x_spA = utterance1.layers[-2].output
    utterance2 = TimeUtterance(input_spB,trainable=False)
    utterance2.load_weights('result/utterance/weights.32-0.42-0.80.h5')
    x_spB = utterance2.layers[-2].output
    """utterance1 = TimeUtterance20(input_spA,trainable=False)
    #utterance1.load_weights('result/utterance20/weights.18-0.51-0.76.h5')
    utterance1.load_weights('result/utterance20/weights.100-0.42-0.80.h5')
    
    utterance2 = TimeUtterance20(input_spB,trainable=False)
    #utterance2.load_weights('result/utterance20/weights.18-0.51-0.76.h5')
    utterance1.load_weights('result/utterance20/weights.100-0.42-0.80.h5')
    x_spA = utterance1.layers[-2].output
    x_spB = utterance2.layers[-2].output"""
    y_spA = Dense(1,activation='sigmoid',name='spA')(x_spA)
    y_spB = Dense(1,activation='sigmoid',name='spB')(x_spB)
    
    input_img = Input(shape=(timestep,96,96,1))
    #gaze = TimeGazeTrain(input_img,trainable=False)
    gaze = TimeFaceTrain(input_img,trainable=False)
    #gaze.load_weights('result/gaze/weights.03-0.16-0.93.h5')
    gaze.load_weights('result/face/weights.20-0.36-0.90.h5')
    img = gaze.layers[-2].output
    y_gaze = Dense(1,activation='sigmoid',name='gaze')(img)
    input_target = Input(shape=(timestep,1))
    input_lang1 = Input(shape=(10,),name='pre_content')
    input_lang2 = Input(shape=(10,),name='content')
    input_lang3 = Input(shape=(1,),name='id1')
    input_lang4 = Input(shape=(1,),name='id2')
    move = moveModel(input_lang1,input_lang2,input_lang3,input_lang4,trainable=False)
    move = move.layers[-2].output
    x = concatenate([x_spA,x_spB,img,input_target])
    #x = Dense(256,activation='relu')(x)
    #x = BatchNormalization()(x)
    #x = Dense(128,activation='relu',
    #              #kernel_regularizer=regularizers.l2(0.01),
    #              #activity_regularizer=regularizers.l2(0.01),
    #              kernel_initializer='random_uniform',
    #               )(x)
    #x = BatchNormalization()(x)
    x = LSTM(128,activation='tanh',recurrent_dropout=0.5,dropout=0.5,
            #kernel_regularizer=regularizers.l2(0.01),
            kernel_initializer='random_uniform', 
            )(x)
    x = BatchNormalization()(x)
    #x = concatenate([x,move])
    x = Dense(64,activation='relu',
                  #kernel_regularizer=regularizers.l2(0.01),
                  #activity_regularizer=regularizers.l2(0.01),
                  kernel_initializer='random_uniform',
                   )(x)
    x = BatchNormalization()(x)
    y = Dense(1,activation='sigmoid',name='passive')(x)
    if not singletask:    
        model = keras.Model(inputs=[input_spA,
                                                    input_spB,
                                                    input_img,
                                                    input_target,
                                                    #input_lang1,
                                                    #input_lang2,
                                                    #input_lang3,
                                                    #input_lang4
                                                   ],
                                                    outputs=[y,y_spA,y_spB,y_gaze]
                                                    )
        #for layer in model.layers[:18]:
        #    layer.trainable = False
        model.compile(loss={'passive':'binary_crossentropy',
                            'spA':'binary_crossentropy',
                            'spB':'binary_crossentropy',
                            "gaze":'binary_crossentropy'},
                      optimizer = keras.optimizers.Adam(1e-04),
                      metrics={'passive':'accuracy',
                                'spA':'accuracy',
                                'spB':'accuracy',
                                "gaze":'accuracy'
                                      },
                      loss_weights = {'passive':1.0,'spA':0.25,'spB':0.25,"gaze":0.25}
                     )
    else:
        model = keras.Model(inputs=[input_spA,
                                                    input_spB,
                                                    input_img,
                                                    input_target,
                                                    #input_lang1,
                                                    #input_lang2,
                                                    #input_lang3,
                                                    #input_lang4
                                                   ],
                                                    outputs=y
                                                    )
        #for layer in model.layers[:18]:
        #    layer.trainable = False
        model.compile(loss='binary_crossentropy',
                      optimizer = keras.optimizers.SGD(1e-02))  
    
    return model
