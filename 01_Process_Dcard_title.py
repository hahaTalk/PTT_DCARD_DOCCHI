#encoding=utf-8
import os
import re
import argparse

emoji_pattern = re.compile("["
    u"\U00010000-\U0010ffff"  # emoji pattern
    u"\u20e3"  # digital
    u"\u2764"  #heart 
    "]+", flags=re.UNICODE)

def process(DCARD_title):
    titles=[]
    labels=[]
    if args.mode == "1" or args.mode == "3":
        for row in (DCARD_title):
            try:
                label,title=row.split('|',2)
                titles.append(title)
                labels.append(label)
            except ValueError:
                continue
    else:                
        for row in (DCARD_title):
            try:
                label,title=row.split('|',2)
                #delete hashtag and emoji pattern
                title = title.replace('<','').replace('>','')
                title = title.replace('#','').replace('＃','')
                title = title.replace('《','').replace('》','')
                title = title.replace('【','').replace('】','')
                title = title.replace('(','').replace(')','')
                title = title.decode('utf8')
                title = emoji_pattern.sub(r'',title)
                title = title.encode('utf8')
                titles.append(title)
                labels.append(label)
            except ValueError:
                continue
    return labels,titles

def write_title_into_file(DCARD_title):
    DCARD_write=open('./dataset/DCARD_title.txt','wb')
    if args.mode == "1" or args.mode == "2":
        labels,titles = process(DCARD_title)
        for item in range(len(titles)):
            DCARD_write.write('%s|%s' %(0,titles[item]))
    else:
        labels,titles = process(DCARD_title)
        for item in range(len(titles)):
            DCARD_write.write('%s|%s' %(labels[item],titles[item]))
    
    
if __name__ == '__main__':
    # Parse command line arguments
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
    DCARD_title = open('./dataset/DCARD_title_0207.txt','r')
    write_title_into_file(DCARD_title)
    
   



