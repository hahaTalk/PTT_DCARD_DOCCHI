#encoding=utf-8
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import utils as np_utils
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Input, LSTM, SpatialDropout1D,Bidirectional,GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, Dropout, BatchNormalization
from keras.layers.convolutional import Conv1D,MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import load_model,Model
from keras import backend as K
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import os.path
import pickle
import os
import numpy as np
import argparse

###global setting###
config = K.tf.ConfigProto()
config.gpu_options.allow_growth = True
session = K.tf.Session(config=config)

DCARD_PTT_title=open('./dataset/DCARD_PTT_title.txt','r')
parser = argparse.ArgumentParser()
parser.add_argument('--mode', required=True,help='1:Oringal dataset  2:Preprocessing dataset  3: Dataset with boards 4: Preprocessing of the dataset with boards')
args = parser.parse_args()    

def parser():
    print('mode : %s' %args.mode)
    if args.mode == "1" : 
        out_cat=2 
        return out_cat
    elif args.mode == "2" : 
        out_cat=2 
        return out_cat
    elif args.mode == "3" : 
        out_cat=12 
        return out_cat
    elif args.mode == "4" : 
        out_cat=12 
        return out_cat
    else: 
        print 'mode only 1 to 4 ' 
        exit
        

def create_data():
    X_data=[]
    y_data=[]

    for row in DCARD_PTT_title:
        try:
            label,text = row.split('|',1)
            X_data.append(text)
            y_data.append(label)
        except ValueError:
            continue

    onehot_encoder = OneHotEncoder(sparse=False)
    y_data = np.array(y_data)
    y_data = y_data.reshape(len(y_data), 1)
    y_data = onehot_encoder.fit_transform(y_data)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test

def token_text(X_train):
    max_seq_len = max([len(txt) for txt in X_train])
    print "Max length %s"  %max_seq_len

    ## Tokenize
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    sequences = tokenizer.texts_to_sequences(X_train)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    X_train = pad_sequences(sequences, maxlen=max_seq_len)

    ###save tokenizer data
    fh = open('tokenizer.pkl','wb')
    pickled_data=pickle.dump(tokenizer,fh)
    
    return X_train,max_seq_len,word_index,tokenizer

### LSTM+CNN model is better for sentiment analyze
### http://konukoii.com/blog/2018/02/19/twitter-sentiment-analysis-using-combined-lstm-cnn-models/
def NNmodel(X_train,y_train,max_seq_len,word_index):
    inp = Input(shape=(max_seq_len,))
    x = Embedding(input_dim=len(word_index)+1, output_dim=128, input_length=max_seq_len, trainable=True)(inp)
    x = SpatialDropout1D(0.3)(x)
    #x = Bidirectional(LSTM(128, return_sequences = True))(x)
    x = LSTM(64, return_sequences = True)(x)
    x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "he_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x_con = concatenate([avg_pool, max_pool])
    x_con = BatchNormalization()(x_con)
    
    predictions = Dense(out_cat,activation="softmax")(x_con)
    
    model = Model(inputs=[inp], outputs=[predictions])
    model.summary()
    model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=0.001 , decay = 0),
                      metrics=['accuracy'])
                      
    #filepath="./model/LSTM_CNN_2pool_v2_Mode%s_{epoch:02d}_{val_acc:.2f}.hdf5" %args.mode
    filepath="./model/best_model_mode_%s" %args.mode
    checkpoint = ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
    callbacks_list=[checkpoint]
    model.fit(X_train,y_train,batch_size=256, epochs=3, validation_split=0.2,callbacks=callbacks_list)
    
    
    return model

if __name__ == '__main__':
    out_cat = parser()
    X_train, X_test, y_train, y_test = create_data()
    X_train,max_seq_len,word_index,tokenizer = token_text(X_train)
    
    if(os.path.exists('./model/best_model_mode_%s.h5' %args.mode) == False):
        model = NNmodel(X_train,y_train,max_seq_len,word_index) 
    else:
        model = load_model('./model/best_model_mode_%s.h5' %args.mode)
    print('number of train : %d' %len(X_train) , 'number of test: %d' %len(X_test))
    model.summary()
        
    X_test = tokenizer.texts_to_sequences(X_test)
    X_test = pad_sequences(X_test, maxlen=max_seq_len)
    predictions = model.predict(X_test,batch_size=256,verbose=1)
    true_ans = np.argmax(y_test,axis=1)
    pred_ans = np.argmax(predictions,axis=1)
    correct_cnt = (np.sum(true_ans == pred_ans))

    print "accuracy: {:2%}".format(float(correct_cnt)/float(len(X_test)))


    
    
    





