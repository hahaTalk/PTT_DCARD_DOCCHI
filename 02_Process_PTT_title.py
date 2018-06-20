#encoding=utf-8
import os
import argparse


def process(PTT_title):
    titles=[]
    labels=[]
    if args.mode == "1" or args.mode == "3":
        for row in (PTT_title):
            try:
                label,title=row.split('|',2)
                titles.append(title)
                labels.append(label)
            except ValueError:
                continue    
    else:
        for row in (PTT_title):
            try:
            #article is replied
                if(len(row.split(':',2)) != 2):
                    label,others = row.split('|',2)
                    others2,title = others.split(']',2)
                    others3,categories=others2.split('[',2)
                    if(categories!='公告'):
                        titles.append(title)
                        labels.append(label)
            except ValueError:        
            #article is deleted
                continue
    return labels,titles

def write_title_into_file(PTT_title):
    PTT_write=open('./dataset/PTT_title.txt','wb')
    labels,titles=process(PTT_title)
    if args.mode== "1" or args.mode == "2":
        for item in range(len(titles)):
            PTT_write.write('%s|%s' %(1,titles[item]))
    else:
        for item in range(len(titles)):
            PTT_write.write('%s|%s' %(labels[item],titles[item]))
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True,help='1:Oringal dataset  2:Preprocessing dataset  3: Dataset with boards 4: Preprocessing of the dataset with boards')
    args = parser.parse_args()    
    if args.mode == "1" : print 'Oringal dataset mode'
    elif args.mode == "2" : print 'Preprocessing dataset mode'
    elif args.mode == "3" : print 'Dataset with boards mode'
    elif args.mode == "4" : print 'Preprocessing of the dataset with boards mode'
    else: 
        print 'mode only 1 to 4 ' 
        exit
    PTT_title = open('./dataset/PTT_title_0207.txt','r')
    write_title_into_file(PTT_title)
