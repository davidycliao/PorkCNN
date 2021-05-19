# PorkCNN  🐖🐖🐖
A Small Project for Pork Barrel Legislation  Classification Using Convolutional Neural Networks 


## Enviroment Setting

- Python 3.8 
- tensorflow==2.5.0
- numpy
- scikit-learn
- scipy 



### Original Trianing Textual Data (Pork Barrel Legislation)

``` 
        text	                                         pork_bill
0	軍人撫卹條例第十八條條文修正草案落實軍人及眷屬之照顧...   1
1	所得稅法第十七條條文修正草案學費之特別扣除額應以每人...   1
2	所得稅法第十一條條文修正草案保險人員申報時得扣除一定...   1
3	土地稅法第二十八條之一條文修正草案土地贈與文教基金會...   1
4	敬老福利生活津貼暫行條例第三條條文修正草案放寬請領資...   1
5	洗錢防制法部分條文修正草案給予法官較大權限；起訴期間...	 0
6	日據時代日本政府國庫券及債券處理條例草案就是保障日據...	 0
7	大陸地區人民來臺從事觀光活動條例草案開放大陸人民觀光...	 0
8	限制欠稅人或欠稅營利事業負責人出境實施條例草案現行法...	 0
9	使用牌照稅法第七條條文修正草案民營汽車駕駛人訓練機構...	 1

```

### Num of Train/ Test Split

```
Num of Train Set: 797 
Not Pork vs Pork: {0: 527, 1: 270}

Num of Test Set: 394 
Not Pork vs Pork: {0: 265, 1: 129}
```

### Model Building & Specification

```
Model: "dcnn"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        multiple                  363800    
_________________________________________________________________
conv1d (Conv1D)              multiple                  40100     
_________________________________________________________________
conv1d_1 (Conv1D)            multiple                  60100     
_________________________________________________________________
conv1d_2 (Conv1D)            multiple                  80100     
_________________________________________________________________
global_max_pooling1d (Global multiple                  0         
_________________________________________________________________
dense (Dense)                multiple                  77056     
_________________________________________________________________
dropout (Dropout)            multiple                  0         
_________________________________________________________________
dense_1 (Dense)              multiple                  257       
=================================================================
Total params: 621,413
Trainable params: 621,413
Non-trainable params: 0
_________________________________________________________________
```



### Evaluation & Classification 

Training conntext: number of pork legislation 792; none-Pork is 399. 

```
                precision    recall  f1-score   support

        Pork       0.92      0.95      0.94       265
    Not Pork       0.90      0.82      0.86       129

    accuracy                           0.91       394
   macro avg       0.91      0.89      0.90       394
weighted avg       0.91      0.91      0.91       394
```
So I need more data to train/test. 😥

```
                     Prediction: Not Pork(0)	   Prediction: Pork(1)
Acutal: Not Pork(0)	                251	                   14
Acutal:    Pork (1)	                 25	                  104
```


### Application on New Dataset (2000 Sampled 6th Legislative Questions)

#### Top 10 of 2000 Samples (more likely to pork barrel)


| Legislator | Pork/Constituency Interest |                               Legislative Questions                          |       Topic      |    Key Word    |
|:----------:|:--------------------------:|:----------------------------------------------------------------------------:|:-----------------|:--------------:|
|陳志彬       | 0.996817171573639      	| 針對公務員85退撫制問題，特向行政院提出質詢。                                        | 公務員退休 ; 退休金 | 公務員 ; 退撫制  |
|丁守中       | 0.995333492755890	        | 針就民眾陳情指出,目前政府對身心障礙者提供之生活津貼,依身心障礙程度等級分為1000元至50...	  | 身心障礙者福利	   |身心障礙者生活津貼  |
|鄭朝明       | 0.983169555664062          | 針對公務人員退休制度相較勞工階級是否符合公平正義，特向行政院提出質詢。	                | 公務員退休	        |公務人員 ; 退休制度|
|林重謨       | 0.974623680114746          | 針對彰化縣教育經費不足,急需中央補助,特向行政院提出緊急質詢。	                        |地方財政 ; 教育經費	 |彰化縣 ; 教育經費 |
|李鎮楠       | 0.973220884799957          | 有鑑於政府有責任讓學童免於飢餓,建請行政院調整現金補助政策,儘速規劃「低收入戶學童營養午餐券...|營養午餐 ; 低收入戶	|現金補助 ; 低收入戶學童營養午餐券|
|彭紹瑾       | 0.922490835189819          | 針對正職勞工之解雇保護制度不足，特向行政院提出質詢。	                              |勞工政策	|解雇 ; 保護制度|
|洪奇昌       | 0.916116058826447          | 針對攸關各醫院健保給付重分配的「醫院支付最適方案」草案，特向行政院提出質詢。	|國民 ; 健康保險	|健保給付 ; 醫院支付最適方案|
|盧秀燕       | 0.902110695838928          | 針對各級農會員工互助會停辦清算，所應給付全體農會員工之離職互助金差額事宜，事關農會員工權益，...	|農會職員 ; 合會	|農會員工互助會 ; 離職互助金差額|
|蔡煌瑯       | 0.897451162338257          |「2005南投花卉嘉年華」活動爭取中央經費補助案由，特向行政院提出質詢。	                    |農業推廣 ; 政府補助	|南投花卉嘉年華 ; 中央經費|
|張碩文       | 0.894091725349426          | 針就軍公教、勞工與漁民都有「老年給付」制度，唯獨農保對於「老年給付」規定付之闕如一事至表關切...	|農保 ; 老年給付	|農保 ; 老年給付|



&nbsp; 

#### Last 10 Rows of 2000 Samples (less likely to pork barrel)


| Legislator | Pork/Constituency Interest |                               Legislative Questions                          |       Topic      |    Key Word    |
|:----------:|:--------------------------:|:----------------------------------------------------------------------------:|:-----------------|:--------------:|
|潘孟安	|0.000000003864951	|就立委及總統選舉在即，立委選舉制度首次採行「單一選區兩票制」，選情緊繃，賄影重重，為免賄選猖...	|選舉風紀	                 |總統選舉 ; 立委選舉 ; 賄選|
|張麗善	|0.000000003287632	|鑒於行政院衛生署雖開放臍帶血移植為常規治療，但迄今仍未訂出合理的收費價格，導致臍帶血市場呈現...	|科技政策 ; 醫療機構 ; 醫療政策	|臍帶血移植 ; 臍帶血銀行 ; 收費標準|
|王幸男	|0.000000002571251	|針對儘管台灣朝野反彈、國際社會指責，中國人大會議仍然在昨天執意通過所謂「反分裂國家法」。這個...	|中國問題 ; 國家政策	       |反分裂國家法 ; 台灣人民 ; 主權|
|陳志彬	|0.000000000511544	|針對層出不窮的黑道暴力討債、卡奴自殺等不幸社會事件，引發自殺潮問題，本席強烈要求主管單位應嚴...	|銀行管理 ; 信用卡 ; 債務	   |暴力討債 ; 卡奴 ; 自殺|
|何智輝	|0.000000000485143	|針對近來信用卡循環利率及現金卡利率均高達18%以上，為存款利率十倍，讓發卡銀行大賺暴利，卻讓...	  |信用卡 ; 利率	               |信用卡循環利率 ; 現金卡利率|
|周守訓	|0.000000000199484	|針對「2005台灣國際影視博覽會」之「台灣影視創投會」活動入選影片評選過程諸多疑點，請新聞局...	   |電影                      	|2005台灣國際影視博覽會 ; 台灣影視創投會|
|林正峰	|0.000000000145981	|針對節目廣告化情節越來越嚴重，國家通訊傳播委員會日前針對播放「竹炭內衣」及「EGF時空膠囊」...	  |電視節目 ; 廣告             |節目廣告化 ; 置入性行銷|
|黃昭順	|0.000000000139926	|為近日部會南遷議題，中央跟地方說詞前後不一，去年北高市長選舉前，陳水扁總統提議將首都南遷，並...	|直轄市 ; 行政區域	           |首都南遷 ; 騙取 ; 選票|
|李復興	|0.000000000103444	|針對行政院為掌握大陸台商財報及陸資來台等動向，從資金、技術與人才外流等三層面，查核台商，並引...	|大陸政策 ; 經濟政策	         |陸資 ; 會計師事務所 ; 查帳|
|高思博	|0.000000000075447	|鑑於「麥肯卡債報告」與行政院金融監督管理委員之卡債協商報告，落差極大，金管會應儘速公布小額信...	|金融管理 ; 債務	            |麥肯卡債報告 ; 卡債協商|


### Use End-to-End Model

```
from tensorflow import keras
model = keras.models.load_model('lour_pork_model') 
```

Step-by-step tutorial finds [here](https://github.com/davidycliao/PorkCNN/blob/main/demo-cnn-pork-barrel-classification-task.ipynb)

## Reference:

- [Yoon Kim, Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
- @gaussic's repo [text-classification-cnn-rnn](https://github.com/gaussic/text-classification-cnn-rnn)
- https://machinelearningmastery.com/start-here/#deeplearning
- Chapter 11, 13, 14 from [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
