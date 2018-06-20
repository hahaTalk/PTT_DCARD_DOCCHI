# PTT_DCARD_DOCCHI
這是中文sentiment analyze的應用，能夠從標題就知道文章是來自PTT還是DCARD  
目前是針對這兩個論壇的六個版(有趣、女孩、美妝、靈異、美食、電影)進行語意分析 

需求
-----------------
[jieba](https://github.com/fxsjy/jieba)   
[keras](https://github.com/keras-team/keras)  
python  

模式
-----------------
這應用分成四個模式
* mode 1 : 只分PTT和DCARD兩個版，且標題完全沒做處理
* mode 2 : 只分PTT和DCARD兩個版，濾掉emoji圖示和部分全型符號
* mode 3 : 分出十二個版，標題完全沒做處理
* mode 4 : 分出十二個版，濾掉emoji圖示和部分全型符號

使用
-----------------
產生處理過的DCARD title檔
```
python 01_Process_Dcard_title.py --mode 4
```
產生處理過的PTT title檔
```
python 02_Process_PTT_title.py --mode 4
```
合併兩個檔案
```
python 03_Cut_and_merge.py
```
訓練
```
python 04_Sentiment_analyze.py --mode 4
```
執行
```
python 05_demo.py
Enter the sentence:從標題就能知道是PTT還是DCARD的文章
 DCARD Funny   31.49%
 DCARD Girl   17.58%
 DCARD Makeup   5.47%
 DCARD Marvel   1.10%
 DCARD Food   0.34%
 DCARD Movie   0.09%
 PTT Womentalk   23.45%
 PTT Makeup   9.08%
 PTT Marvel   2.39%
 PTT Food   0.04%
 PTT Movie   0.73%
 PTT Joke   8.22%
這篇文章有 31.49% 的機率是來自 DCARD Funny
```


Pre-train model  
https://drive.google.com/drive/folders/14FZEHu7p2DU0spWvVajcX6SmaMKmRha8


