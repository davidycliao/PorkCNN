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
Num of Train Set: 2693  
Not Pork vs Pork: {0: 1741, 1: 952}

Num of Test Set: 1327 
Not Pork vs Pork: {0: 861, 1: 466}
```

### Model Building & Specification

```
Model: "dcnn"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        multiple                  1400000   
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
Total params: 1,657,613
Trainable params: 1,657,613
Non-trainable params: 0
_________________________________________________________________
```



### Evaluation & Classification 

Training conntext: Number of Pork Legislation 1418; Number of None-Pork Legislation is 2602. 


```
              precision    recall  f1-score   support

           0       0.97      0.98      0.97       861
           1       0.97      0.94      0.95       466

    accuracy                           0.97      1327
   macro avg       0.97      0.96      0.96      1327
weighted avg       0.97      0.97      0.97      1327
```

```
                     Prediction: Not Pork(0)	   Prediction: Pork(1)
Acutal: Not Pork(0)	                846	                   15
Acutal:    Pork (1)	                 30	                  436
```


### Application on New Dataset (2000 Sampled 6th Legislative Questions)

#### Top 10 of 2000 Samples (more likely to pork barrel)


| Legislator | Pork/Constituency Interest |                               Legislative Questions                          |       Topic      |    Key Word    |
|:----------:|:--------------------------:|:----------------------------------------------------------------------------:|:-----------------|:--------------:|
|賴士葆	|0.997264862060547	|對行政院所屬勞工保險局發放有關福利敬老津貼有失周詳，不合公道一案，特向行政院提出質詢。	      |老人福利	                    |勞工保險局 ; 福利敬老津貼|
|邱鏡淳	|0.996278405189514	|就政府為減緩政府財政負擔及維持退休人員與現職人員權益的平衡，推行「教育人員退休所得合理化方案...	|教育人員退休 ; 退休金          |教育人員 ; 退休所得|
|林鴻池	|0.996166765689850	|針對目前政府十三萬臨時人員已依勞退新制提撥退休金，但仍尚未納入勞基法保障，政府應儘速公布將臨...	|勞動基準 ; 聘僱人員 ; 法律適用範圍 |臨時人員 ; 勞基法|
|林正峰	|0.995356976985931	|針對政府準備修法推動「二代健保」，健保保費採取「年度所得總額」為計算基礎，而非採用扣除免稅額...	|國民 ; 健康保險 ; 保險費|二代健保 ; 年度所得總額|
|賴清德	|0.990603744983673	|為改善低所得家庭就業困難，提昇其工作所得，使脫離貧窮困境，特向行政院提出質詢。	             |低收入戶 ; 就業 | 低所得家庭 ; 就業困難|
|林重謨	|0.989980101585388      |針對財政部關於遺產及贈與稅制之改革一事，特向行政院提出質詢。                                    |遺產稅 ; 贈與稅 ; 賦稅改革 | 遺產稅 ; 贈與稅|
|林重謨	|0.989181876182556	|針對國防部當初為了因應政府取消軍教免稅優惠，將志願役軍士官勤務加給自八十九年七月起調高，平均...	|軍人 ; 教育人員 ; 課稅|軍教課稅 |
|林建榮	|0.988365650177002	|為建請回復「漁業動力用油優惠標準」百分之二十八之補助措施，以扶助弱勢漁民生計，特向行政院提出質詢。    |漁業補助 | 漁業動力用油優惠標準|
|羅志明	|0.986930191516876	|針對為照顧弱勢農民，高雄市及各地方政府為都市城鄉發展之均衡，高雄市的都市周邊區域之小港區應比...	|農業補助 |	小港區 ; 重工業回饋地方基金 ; 資助農民|
|費鴻泰	|0.986919820308685	|針對政府公教貸款利率高於一般商業銀行專案辦理之房屋貸款利率，使公教人員之房貸利息負擔沈重依舊...	|公務員福利 ; 房屋貸款 ; 利率|公教貸款利率|





&nbsp; 

#### Last 10 Rows of 2000 Samples (less likely to pork barrel)


| Legislator | Pork/Constituency Interest |                               Legislative Questions                          |       Topic      |    Key Word    |
|:----------:|:--------------------------:|:----------------------------------------------------------------------------:|:-----------------|:--------------:|
|高思博	|0.000012218362826	|鑑於「麥肯卡債報告」與行政院金融監督管理委員之卡債協商報告，落差極大，金管會應儘速公布小額信...	|金融管理 ; 債務	|麥肯卡債報告 ; 卡債協商|
|羅世雄	|0.000011173857274	|針對若干不肖業者以「課程大放送」方式，規避定型化契約規範，欺瞞消費者，爰此，主管單位應提出立...	|語言 ; 補習班 ; 消費者保護	|定型化契約 ; 語言補習班|
|陳啟昱	|0.000007897281648	|針對高雄港港區監視系統，得標廠商與港務局就監視系統規格有所爭議，經行政院公共工程委員會調解，...	|商港 ; 工程招標 ; 電子監視	|高雄港 ; 監視系統|
|鄭朝明	|0.000007136633940	|有鑒於近年消費金融債務糾紛頻傳，部分討債公司採取討債手法過於激烈，其中包括潑灑油漆、半夜恐嚇...	|金融管理 ; 債務 ; 暴力	|金融債務 ; 討債公司|
|吳敦義	|0.000006591444617	|有鑑於福建省政府委員之職務係在監督縣自治事項及辦理其他行政院交辦事項，然現任福建省政府委員、...	|政府官員 ; 行政中立	|福建省政府委員 ; 陳滄江 ; 拉票|
|柯淑敏	|0.000005787265309	|針對台灣已進入金控年代，但目前主管機關仍然是以分業的法規來監理金控旗下各子公司的轉投資，造成...	|金融業 ; 金融管理	|金控 ; 子公司 ; 轉投資|
|陳瑩	     |0.000005518909347	 |有鑑於國內監聽情況嚴重氾濫，直接侵犯人民隱私權，為維護基本人權以及控管監聽行為獨立性，監聽審...	|竊聽 ; 人權 |監聽 ; 基本人權|
|林進興	|0.000004838947007	|針對行政院金融監督管理委員會為了解決國人廣大的卡債問題，積極推廣債務協商機制，政策本意良好，...	|信用卡 ; 債務	|卡債 ; 協商機制|
|紀國棟	|0.000003577750704	|鑑於監視器是維持治安不可或缺之工具，惟監視器維修經費龐大，若遇颱風豪雨嚴重損壞費用很高，多數...	|電子監視	|監視器.損壞維修|
|王昱婷	|0.000002748315183	|針對我國各產險業者長期有聯合壟斷之情事發生，嚴重影響臺灣社會消費者之權益。惟礙於現行法令，如...	|金融管理 ; 保險業	|產險業 ; 聯合壟斷|




### Use End-to-End Model

```
from tensorflow import keras
model = keras.models.load_model('lour_pork_model') 
```

Step-by-step tutorial finds [here](https://github.com/davidycliao/PorkCNN/blob/main/demo-cnn-pork-barrel-classification-task.ipynb)

## Reference:

- [Yoon Kim, Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
- @gaussic's repo [text-classification-cnn-rnn](https://github.com/gaussic/text-classification-cnn-rnn)
- The collection of legislation was manually labelled by  Profession Luor, Ching-Jyuhn  and his research team.  We appreciate the assistance in providing the data.
- Chapter 11, 13, 14 from [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
