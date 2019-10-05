from keras_model import *
import json,pickle
import re
import MeCab
import numpy as np
import glob
from collections import Counter
from sklearn.model_selection import train_test_split,KFold
import keras
from keras.layers import *
import keras.backend as K
from keras_self_attention import SeqSelfAttention

saleVal = re.compile('[:.0-9a-z_A-Z!?%()|]')
def clean_sentence(text:str) -> str :
    """
    """
    #text = re.sub(r"<笑>",'',text)
    #text = "(D_ウズ)でもそれ鑑賞の一部だからしようと思っているんだけど:"
    text = saleVal.sub("",text)
    text = re.sub(r"(\()(.*?)\)",'',text)
    text = re.sub(r"<.?>",'',text)
    return text

usr2id = {'A:':0, 'B:':1, 'C:':2}
label2id = {'s':0, 'b':1, 'f':2, 'r':3, 'i':4}

# def get_feature(text_files :list ,id_list :list) -> list:
#     """
#     """
#     sentences = []
#     label = []
#     users = []
#     for i in id_list:
#         try:
#             with open(text_files[i],encoding='shift-jis') as f:
#                 lines = f.readlines()
#         except:
#             with open(text_files[i],encoding='utf-8') as f:
#                 lines = f.readlines()

#         lines = [line.rstrip().split() for line in lines]
#         for line in lines:
#             #if line[-1] == 'f':continue
#             text = clean_sentence(line[-2])
#             if text == "" or text in ["<笑>","<息>" ,"<声>"]:continue
#             label.append(label2id[line[-1]])
#             sentences.append(text)
#             users.append(line[-3])

#     users = get_usrID(users)

#     return sentences, np.array(users,dtype=np.float32), np.array(label,dtype=np.int32)

# def get_usrID(users :list) -> list:
#     """ 
#     直前の発話に対して，
#         発話者が変わった場合 1
#         発話者が同じ場合は 0
#     :param: users : list 各発話の発話者(A,B,C)
#     return: userID : list 発話者の変化(0,1)
#     """
#     userID = [0 if users[i] == users[i+1] else 1 for i in range(len(users)-1)]
#     return [0] + userID

# def wakati(sentences :list) -> list:
#     """
#     分かち書きして返す関数
#     param: [私は学生です,私は犬です]
#     return: [[私 は 学生 です],[私 は 犬 です]]
#     """
#     texts = []
#     m = MeCab.Tagger("-d /usr/lib/mecab/dic/mecab-ipadic-neologd")
#     for sentence in sentences:
#         word_list = list()
#         words_chasen = m.parse(sentence).split('\n')
#         for word_chasen in words_chasen:
#             """
#             ['映画', '名詞,一般,*,*,*,*,映画,エイガ,エイガ']
            
#             """
#             if word_chasen == 'EOS': break
#             tag1 = word_chasen.split("\t")[1].split(',')[0]#映画
#             tag2 = word_chasen.split("\t")[1].split(',')[1]#名詞
            
#             if tag1 not in word2tag:
#                 word2tag[tag1] = len(word2tag)
#             if tag2 not in word2tag2:
#                 word2tag2[tag2] = len(word2tag2)
                
#             #word = word_chasen.split("\t")[0]
#             word_list.append(word_chasen.split("\t")[0])
#             #word_list.append((word_chasen.split("\t")[0],word2tag[tag1],word2tag2[tag2])
#         #word_list = ' '.join(word_list)
#         texts.append(' '.join(word_list))
#     return texts

# def get_pos(sentences :list, MAX_SIZE :int = 10) -> list:
#     """
#     分かち書きして返す関数
#     param: [私は学生です,私は犬です]
#     return: [[私 は 学生 です],[私 は 犬 です]]
#     """
#     texts = []
#     word2tag={}
#     word2tag2={}
#     tag_list=[]
#     tag_list2=[]
    
#     m = MeCab.Tagger("-d /usr/lib/mecab/dic/mecab-ipadic-neologd")
#     for sentence in sentences:
#         word_list = list()
#         tags=[0]*MAX_SIZE
#         tags2=[0]*MAX_SIZE
#         words_chasen = m.parse(sentence).split('\n')
#         for i,word_chasen in enumerate(words_chasen):
#             """
#             ['映画', '名詞,一般,*,*,*,*,映画,エイガ,エイガ']
            
#             """
#             if word_chasen == 'EOS' or i ==MAX_SIZE: break
#             tag1 = word_chasen.split("\t")[1].split(',')[0]#名詞
#             tag2 = word_chasen.split("\t")[1].split(',')[1]#一般
#             if tag1 not in word2tag:
#                 word2tag[tag1] = len(word2tag)+1
#             if tag2 not in word2tag2:
#                 word2tag2[tag2] = len(word2tag2)+1
#             tags[i]=word2tag[tag1]
#             tags2[i]=word2tag2[tag2]
#         tag_list.append(tags)
#         tag_list2.append(tags2)
    
            
#     return word2tag, word2tag2,np.array(tag_list), np.array(tag_list2)

def make_Embedding(model, tokenizer, EMBEDDING_DIM=300) -> list:
    """
    重み行列作成
    """
    embedding_metrix = np.zeros((len(tokenizer.word_index)+1,EMBEDDING_DIM))
    for word, i in tokenizer.word_index.items():
        try:
            embedding_vector = model[word]
            embedding_metrix[i] = embedding_vector
        except KeyError:#学習済みモデルにない単語は，乱数で値を決定
            #print(word)
            embedding_metrix[i] = np.random.normal(0, np.sqrt(0.25), EMBEDDING_DIM)
    return embedding_metrix

def save_word2id(tokenizer, INPUT=''):
    """
    単語→idの辞書を保存
    """
    import json
    f = open(INPUT,'w')#.json
    json.dump(tokenizer.word_index,f)
    f.close()

def save_EmbeddingMetrix(embedding_metrix, INPUT=''):
    """
    分散行列の保存@python3
    """
    f = open(INPUT,"wb")#
    pickle.dump(embedding_metrix,f,protocol=2)
    f.close()


def word2id(x :list ,MAX_SIZE :int = 10) -> list:
    """
    param: x...分かち書きされた１文章
    return: t...分かち書きされた各単語をidに変換したもの
    """

    
    t = np.zeros(MAX_SIZE)
    for i,word in enumerate(x[-MAX_SIZE:]):
        word = word.lower()
        if word in filters: continue
        try:
            t[i] = word_index[word]
        except:
            #print(word,end=' ')
            t[i] = 0
    t = np.array(t).astype('int32')
    return t 

##
def clean_sentence(text:str) -> str :
    """
    """
    import re
    #text = re.sub(r"<笑>",'',text)
    #text = "(D_ウズ)でもそれ鑑賞の一部だからしようと思っているんだけど:"
    text = saleVal.sub("",text)
    text = re.sub(r"(\()(.*?)\)",'',text)
    text = re.sub(r"<.?>",'',text)
    return text

usr2id = {'A:':0, 'B:':1, 'C:':2}
label2id = {'s':0, 'b':1, 'f':2, 'r':3, 'i':4}

def get_usrID(users :list) -> np.ndarray:
    """
    直前の発話に対して，
        発話者が変わった場合 1
        発話者が同じ場合は 0
    :param: users : list 各発話の発話者(A,B,C)
    return: userID : list 発話者の変化(0,1)
    """
    userID = [0 if users[i] == users[i+1] else 1 for i in range(len(users)-1)]
    return np.array([0] + userID)

def encode_sentence(text_list:list,  MAX_SIZE: int = 10) :
    """

    :param text_list:
    :return: sentences 単語をIDに変換したベクトル
             X_tag     単語の品詞をone-hot-eoncodingしたベクトル
             X_tag2    単語の品詞<詳細>をone-hot-eoncodingしたベクトル

    """
    label, users, sentences = [],[],[]
    for i in range(len(text_list)):
        try:
            with open(text_list[i], encoding='shift-jis') as f:
                lines = f.readlines()
        except:
            with open(text_list[i], encoding='utf-8') as f:
                lines = f.readlines()

        lines = [line.rstrip().split() for line in lines]
        for line in lines:
            text = clean_sentence(line[-2])
            if text == "" or text in ["<笑>", "<息>", "<声>"]: continue
            label.append(label2id[line[-1]])
            sentences.append(text)
            users.append(line[-3])

    users = get_usrID(users)
    feature , X_tag, X_tag2 = wakati(sentences, MAX_SIZE= MAX_SIZE)

    return feature, X_tag, X_tag2, users, np.array(label, dtype=np.int32)


def wakati(sentences: list , MAX_SIZE: int = 10):
    """
    分かち書きして返す関数
    param: [私は学生です,私は犬です]
    return:
    """
    texts = []
    word2tag = {}
    word2tag2 = {}
    tag_list = []
    tag_list2 = []


    m = MeCab.Tagger("-d /usr/lib/mecab/dic/mecab-ipadic-neologd")
    for sentence in sentences:

        tags = [0] * MAX_SIZE
        tags2 = [0] * MAX_SIZE
        word_list = ["0"] * MAX_SIZE
        words_chasen = m.parse(sentence).split('\n')
        for i, word_chasen in enumerate(words_chasen[-MAX_SIZE:]):
            """
            ['映画', '名詞,一般,*,*,*,*,映画,エイガ,エイガ']
            """
            if word_chasen == 'EOS': break
            tag1 = word_chasen.split("\t")[1].split(',')[0]  # 映画
            tag2 = word_chasen.split("\t")[1].split(',')[1]  # 名詞

            if tag1 not in word2tag:
                word2tag[tag1] = len(word2tag)
            if tag2 not in word2tag2:
                word2tag2[tag2] = len(word2tag2)

            # word = word_chasen.split("\t")[0]
            word_list[i] = word_chasen.split("\t")[0]

            tags[i] = word2tag[tag1]
            tags2[i] = word2tag2[tag2]
        tag_list.append(tags)
        tag_list2.append(tags2)
        texts.append(word_list)

    feature = [word2id(words, MAX_SIZE=MAX_SIZE) for words in texts]

    return np.array(feature), np.array([np.eye(len(word2tag))[i] for i in tag_list]),\
           np.array([np.eye(len(word2tag2))[i] for i in tag_list2])


def word2id(x: list, MAX_SIZE: int = 10) -> list:
    """
    param: x...分かち書きされた１文章
    return: t...分かち書きされた各単語をidに変換したもの
    """
    import json
    #with open('/mnt/aoni02/katayama/new_word_index.json') as w:
    with open('/mnt/aoni02/katayama/chainer/keras/word_index.json') as w:
        word_index = json.load(w)
    
    t = np.zeros(MAX_SIZE)
    for i, word in enumerate(x):
        #print('word',word)
        word = word.lower()
        if word in filters: continue
        try:
            t[i] = word_index[word]
        except:
            if word != '0': print(word,end=' ') 
            t[i] = 0
    t = np.array(t).astype('int32')
    return t
##

#ここから学習
with open('/mnt/aoni02/katayama/new_embedding_metrix.pkl',"rb") as f:
    embedding_metrix = pickle.load(f)
filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
with open('/mnt/aoni02/katayama/new_word_index.json') as w:
    word_index = json.load(w)

def moveModel(i1 = Input(shape=(10,)),i2 = Input(shape=(10,)),u1_id = Input(shape=(1,)),u2_id = Input(shape=(1,)),trainable=True):
    
    Embed = keras.layers.Embedding((embedding_metrix.shape[0]), 300,mask_zero = True,weights = [embedding_metrix],input_length=10,trainable=False)
    x_a = Embed(i1)
    x_b = Embed(i2)
    
    shareLSTM = LSTM(64,return_sequences=False,dropout=0.5,recurrent_dropout=0.5,name='lstm',trainable=trainable)
    #shareSelfAtt = SeqSelfAttention(units=32,name='self-att',trainable=trainable)
    shareSum = Lambda(lambda a: K.sum(a,axis=-2), output_shape=(64,))
    shareDense = Dense(16,activation='relu',name='dense',trainable=True)
    shareDropout = Dropout(0.5)
    for f in [shareLSTM,shareDense,shareDropout]:
        x_a = f(x_a)
        x_b = f(x_b)
    x = concatenate([x_a,x_b,u1_id],axis=-1)
    o = Dense(5,activation='softmax')(x)
    m = keras.Model([i1,i2,u1_id],o)
    m.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'],optimizer=keras.optimizers.Adam(1e-03))
    #m.load_weights('../keras/move0910.h5',by_name=True)
    
    return m

def moveModel(i1 = Input(shape=(2,10,)),t1 = Input(shape=(2,10,2)),u1_id = Input(shape=(1,)),trainable=True):
    
    Embed = Embedding((embedding_metrix.shape[0]), 300,mask_zero = True,weights = [embedding_metrix],input_length=10,trainable=False)
    x_a = TimeDistributed(Embed)(i1)
    x_a = TimeDistributed(Dense(64,activation='relu'))(x_a)
    x_a = concatenate([x_a,t1],axis=-1)
    shareLSTM = LSTM(64,return_sequences=False,dropout=0.5,recurrent_dropout=0.5,name='lstm',trainable=trainable)
    shareSum = Lambda(lambda a: K.sum(a,axis=-2), output_shape=(64,))
    shareDense = Dense(16,activation='relu',name='dense',trainable=True)
    shareDropout = Dropout(0.5)
    for f in [shareLSTM]:#,shareDense,shareDropout]:
        x_a = TimeDistributed(f)(x_a)
    #xt = TimeDistributed(LSTM(2))(t1)
    #x = concatenate([x_a,xt],axis=-1)
    x = LSTM(64,dropout=0.5)(x_a)
    x = BatchNormalization()(x)
    x = concatenate([x,u1_id])
    o = Dense(5,activation='softmax')(x)
    model = keras.Model([i1,t1,u1_id],o)
    model.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'],#optimizer=keras.optimizers.SGD(0.1))
                  optimizer=keras.optimizers.Adam(1e-03))
    return model

def moveModel(i1 = Input(shape=(20,)),p1 = Input(shape=(20,13)),p2 = Input(shape=(20,30)),u1_id = Input(shape=(1,)),trainable=True):
    
    Embed = Embedding((embedding_metrix.shape[0]), 300,mask_zero = True,weights = [embedding_metrix],input_length=20,trainable=False)
    x_a = Embed(i1)
    x_a = Dense(64,activation='relu')(x_a)
    x_p1 = Dense(6,activation='relu')(p1)
    x_p2 = Dense(6,activation='relu')(p2)
    x_a = concatenate([x_a,x_p1, x_p2],axis=-1)
    shareLSTM = LSTM(64,return_sequences=False,dropout=0.5,recurrent_dropout=0.5,name='lstm',trainable=trainable)
    shareSum = Lambda(lambda a: K.sum(a,axis=-2), output_shape=(64,))
    shareDense = Dense(16,activation='relu',name='dense',trainable=True)
    shareDropout = Dropout(0.5)
    for f in [shareLSTM]:#,shareDense,shareDropout]:
        x_a = f(x_a)
    x = BatchNormalization()(x_a)
    x = concatenate([x,u1_id])
    o = Dense(5,activation='softmax')(x)
    model = keras.Model([i1, p1, p2, u1_id],o)
    model.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'],#optimizer=keras.optimizers.SGD(0.1))
                  optimizer=keras.optimizers.Adam(1e-04))
    return model

if __name__ == '__main__':
    text_files = glob.glob('/mnt/aoni02/katayama/dataset/Trans_withLabel/*txt')
    with open('./word_index.json') as w:
        word_index = json.load(w)
    
    #feature, X_pos, X_pos2, X_id, label = encode_sentence(text_files)
    #print(np.shape(feature),np.shape(X_pos),np.shape(X_pos2),np.shape(X_id),np.shape(label))
    #print(1/0)
    X , X_id, y= get_feature(text_files,list(range(len(text_files))))
    word2tag, word2tag2,tag_list, tag_list2 = get_pos(X,MAX_SIZE=20)
    feature = np.array([word2id(w.split(),MAX_SIZE=20) for w in wakati(X)])
    window_size = 1
    #X_pos = np.concatenate([tag_list.reshape(len(tag_list),1,10), tag_list2.reshape(len(tag_list2),1,10)],axis=1)
    print(np.shape(tag_list),np.shape(tag_list2))
    tag_list = np.array([np.eye(len(word2tag)+1)[i] for i in tag_list])
    tag_list2 = np.array([np.eye(len(word2tag2)+1)[i] for i in tag_list2])
    print(np.shape(tag_list),np.shape(tag_list2))
    #X_pos = np.concatenate([tag_list, tag_list2],axis=1)
    #X_pos = np.concatenate([tag_list.reshape(len(tag_list),10,1), tag_list2.reshape(len(tag_list2),10,1)],axis=-1)
    #print(X_pos.shape)
    print(feature.shape)
    X1 = feature[:-window_size]
    #X2 = feature[1:]
    #X_pos = np.array([X_pos[i:i+window_size+1] for i in range(0,len(feature)-window_size)])
    #X = np.array([feature[i:i+window_size+1] for i in range(0,len(feature)-window_size)])
    #X_id = np.array([X_id[i:i+window_size] for i in range(0,len(feature)-window_size)]).reshape(-1,window_size)
    #y = y[window_size:]
    print(len(feature)==len(y)==len(X_id)==len(tag_list)==len(tag_list2))
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    class_weight = {0:1.0, 1:1.0, 2:6.0, 3:.0, 4:3.0}
    for i in range(5):
        class_weight[i] = min (len(y) / Counter(y)[i], 15.0 )
    print("class_weight",class_weight)
    Acc=[]
    
    for i in range(5):
        X_train, X_val, y_train, y_val = train_test_split(feature,y,test_size=0.2,random_state=i*2+1)
        Xp_train, Xp_val, y_train, y_val = train_test_split(tag_list,y,test_size=0.2,random_state=i*2+1)
        Xp2_train, Xp2_val, y_train, y_val = train_test_split(tag_list2,y,test_size=0.2,random_state=i*2+1)
        #X1_train, X1_val, y_train, y_val = train_test_split(X1,y,test_size=0.2,random_state=i*2+1)
        #X2_train, X2_val, y_train, y_val = train_test_split(X2,y,test_size=0.2,random_state=i*2+1)
        X_id_train, X_id_val, y_train, y_val = train_test_split(X_id,y,test_size=0.2,random_state=i*2+1)
        """
        model = moveModel(i1 = Input(shape=(2,10,)),
                                       #i2 = Input(shape=(20,)),
                                       u1_id = Input(shape=(1,)),
                                       #u2_id = Input(shape=(1,)),
                                       t1 = Input(shape=(2,10,2)),
                                       trainable=True)
        """
        model = moveModel()
        #model.summary()

        hist = model.fit(
                        [
                         X_train,
                         Xp_train,
                         Xp2_train,
                         #X1_train,
                         #X2_train,
                         X_id_train],
                         y_train,
                         epochs=200,batch_size=128,validation_split=0.2,
                         validation_data=([
                                           X_val,
                                           Xp_val,
                                           Xp2_val,
                                           #X1_val,
                                           #X2_val,
                                           X_id_val],
                                          y_val),
                         callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,verbose=1)],
                         verbose=0,
                         class_weight = class_weight
                        )
        #acc = model.evaluate([X1_val,X2_val,X_id_val],y_val)[1]
        acc = model.evaluate([X_val,
                              Xp_val,
                              Xp2_val,
                              X_id_val
                             ],y_val)[1]
        print('acc',acc)
        Acc.append(acc)
            
    print(np.average(Acc))
    pred = model.predict([X_val,Xp_val,Xp2_val,X_id_val])
    pred = [np.argmax(i) for i in pred]
    print(confusion_matrix(y_val,pred))