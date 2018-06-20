#encoding=utf-8
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import load_model
import jieba
import pickle
import os
import numpy as np
#import uniout

s=''
sentence=[]
jieba.set_dictionary('./jieba_extra_dict/dict.txt.big')

Board_txt = open('Board_label.txt','r')
titles=[]
labels=[]
for row in Board_txt:
    try:
        label,title = row.split('|',1)
        title = title.replace('\n',' ')
        titles.append(title)
        labels.append(label)
    except ValueError:
        continue

fh=open('tokenizer.pkl','rb')
tokenizer=pickle.load(fh)    
model = load_model('./model/best_model_mode_4.h5')

while 1:
    data=raw_input("Enter the sentence:")
    words = jieba.cut(data, cut_all=False)
    for word in words:
        s=s+word+' '
    sentence.append(s.encode('utf8'))
    sequential = tokenizer.texts_to_sequences(sentence)
    indata = pad_sequences(sequential, maxlen=1750)
    predictions = model.predict(indata)
    predictions = predictions.round(4)
    pred_ans = np.argmax(predictions,axis=1)

    for i in range(len(titles)):
        print("%s  %.2f%%" %(titles[i] , predictions[:,i]*100))
        
    print ('這篇文章有 %.2f%% 的機率是來自%s' %(predictions[:,pred_ans[0]]*100,titles[pred_ans[0]]))
    s=''
    del sentence[:]

