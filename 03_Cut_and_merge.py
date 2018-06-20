#encoding=utf-8
import os
import jieba
import argparse


def segment():
    
    DCARD_title = open('./dataset/DCARD_title.txt','r').read()
    PTT_title = open('./dataset/PTT_title.txt','r').read()
    DCARD_PTT_title = open('./dataset/DCARD_PTT_title.txt','wb')

    DCARD_words = jieba.cut(DCARD_title, cut_all=False)
    PTT_words = jieba.cut(PTT_title, cut_all=False)

    for word in DCARD_words:
        DCARD_PTT_title.write((word + ' ').encode('utf8'))
    for word in PTT_words:
        DCARD_PTT_title.write((word + ' ').encode('utf8'))


if __name__=='__main__':
    jieba.set_dictionary('./jieba_extra_dict/dict.txt.big')
    segment()
