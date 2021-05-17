# PorkCNN   🐖🐖🐖🐖🐖🐖🐖🐖🐖🐖🐖🐖🐖🐖🐖🐖🐖🐖🐖🐖🐖🐖🐖🐖🐖
A Small Project for Pork Barrel Legislation  Classification Using CNN 


## Enviroment Setting

- Python 3.8 
- tensorflow==2.5.0
- numpy
- scikit-learn
- scipy 



### Original Trianing Textual Data 

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
Shape of X Train: (797, 123) 
Shape of X Test:  (394, 123) 
Shape of Y Trian: (797,) 
Shape of Y Test:  (394,)
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
Predictions:              Not Pork(0)	 Predictions:Pork(1)
Acutal: Not Pork(0)	         251	                 14
Acutal:    Pork (1)	          25	                104
```


### Application on New Dataset (200 Sampled Legislative Questions)

#### Top 5 of 200 Samples

| Pork/Constituency Interest |      Legislative Questions   |  Topic |  Key Word |
|:----------:|:-------------:|:------:|:------:|
|0.811236739158630|	針對政府公教貸款利率高於一般商業銀行專案辦理之房屋貸款利率，使公教人員之房貸利息負擔沈重依舊...|	公務員福利 ; 房屋貸款 ; 利率|	公教貸款利率|
|0.443107038736343|	南投縣為農業縣份，工商不發達，稅收財源不豐裕，全年度自有財源收入尚不敷支應全縣人事費支出，而...|	地方財政|	稅收 ; 地方財政
0.312157511711121 |	針對目前政府十三萬臨時人員已依勞退新制提撥退休金，但仍尚未納入勞基法保障，政府應儘速公布將臨...|	勞動基準 ; 聘僱人員 ; 法律適用範圍|	臨時人員 ; 勞基法|
|0.247517079114914|	針對近年來財務持續惡化的中央健康保險局，利用健保費給付銀行巨額借款利息表示斥責。健保局用民眾...|	國民 ; 健康保險 ; 保險費|	財務 ; 中央健康保險局 ; 健保費給付|
|0.157373845577240|	針對交通部修正「道路交通安全規則」，將大客車、聯結車、大貨車等大型車職業司機考照執業年齡上限...|	退休年齡|	道路交通安全規則 ; 職業司機 ; 退休年齡|


&nbsp; 

#### Last 10 Rows of 200 Samples

| Pork/Constituency Interest |      Legislative Questions   |  Topic |  Key Word |
|:----------:|:-------------:|:------:|:------:|
|0.000001753699053|	針對行政院金融監督管理委員會（金管會）宣稱，台灣上市上櫃公司投資中國累計匯回資金比例達7.9...|	大陸政策 ; 對外投資	| 投資中國 ; 台商資金匯回|
|0.000001426775611|	就立委及總統選舉在即，立委選舉制度首次採行「單一選區兩票制」，選情緊繃，賄影重重，為免賄選猖...|	選舉風紀|	總統選舉 ; 立委選舉 ; 賄選|
|0.000001366255105|	針對政府若干公務人員涉嫌重大弊案，諸如高雄捷運弊案、股市禿鷹案、國道電子收費（ETC）案，嚴...|	公務員 ; 行政中立 ; 黨政關係 ; 政府官員|	公務人員 ; 弊案 ; 政務官 ; 行政中立 ; 輔選|
|0.000001024151629|	針對行政院金融監督管理委員會提出「現金卡廣告最新規範」，宣布自五月一日起，現金卡應暫停在電子...|	消費貸款 ; 銀行管理 | 現金卡廣告|
|0.000001019861088|	針對詐騙集團猖獗，行騙手法推陳出新，令人防不勝防，據統計約有八成民眾曾接獲詐騙集團的行騙電話...|	電話 ; 詐欺|	詐騙集團 ; 電話 ; 簡訊|

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
