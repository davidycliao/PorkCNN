# PorkCNN  🐖🐖🐖
A Small Project for Pork Barrel Legislation  Classification Using Convolutional Neural Networks 

I have trained a pork-barrel classifier on the human-labelling introduction of bill and legislation from 2004-2012 (provided by Dr Ching-Jyuhn Luor, National Taipei University). The pre-trained model is available on my GitHub repo for end-to-end use. If there’s anything you need about the application, please don’t hesitate to send me a message.

## Enviroment Setting

- Python 3.8 
- tensorflow==2.5.0
- numpy
- scikit-learn
- scipy 



### Original Trianing  Data (Pork Barrel Legislation from 2004-2012)

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
Num of Train Set: 4852 
Not Pork vs Pork: {0: 3167, 1: 1685}

Num of Test Set: 2391 
Not Pork vs Pork: {0: 1566, 1: 825}
```

### Model Building & Specification

```
Model: "dcnn"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        multiple                  586600    
_________________________________________________________________
conv1d (Conv1D)              multiple                  40100     
_________________________________________________________________
conv1d_1 (Conv1D)            multiple                  60100     
_________________________________________________________________
conv1d_2 (Conv1D)            multiple                  80100     
_________________________________________________________________
conv1d_3 (Conv1D)            multiple                  100100    
_________________________________________________________________
conv1d_4 (Conv1D)            multiple                  0 (unused)
_________________________________________________________________
global_max_pooling1d (Global multiple                  0         
_________________________________________________________________
dense (Dense)                multiple                  128256    
_________________________________________________________________
dropout (Dropout)            multiple                  0         
_________________________________________________________________
dense_1 (Dense)              multiple                  257       
=================================================================
Total params: 995,513
Trainable params: 995,513
Non-trainable params: 0
_________________________________________________________________
```



### Evaluation & Classification 

Training conntext: Number of Pork Legislation 2510; Number of None-Pork Legislation is 4733. 


```
              precision    recall  f1-score   support

           0       0.95      0.97      0.96      1566
           1       0.94      0.91      0.92       825

    accuracy                           0.95      2391
   macro avg       0.95      0.94      0.94      2391
weighted avg       0.95      0.95      0.95      2391
```

```
                     Prediction: Not Pork(0)	   Prediction: Pork(1)
Acutal: Not Pork(0)	               1513	                   53
Acutal:    Pork (1)	                 68	                  757
```


### Application on New Dataset (2000 Sampled 6th Legislative Questions)

#### Top 10 of 2000 Samples (more likely to pork barrel)


| Legislator | Pork/Constituency Interest |                               Legislative Questions                          |       Topic      |    Key Word    |
|:----------:|:--------------------------:|:----------------------------------------------------------------------------:|:-----------------|:--------------:|
| 陳啟昱	   |0.996769189834595	        |鑑於現行《所得稅法》第十七條規定特別扣除額教育支出部分，僅以納稅義務人之子女就讀大專院校為限...|	Income tax; education expenses; deductions | Income Tax Law; Special Deductions; Educational Expenditure|
| 林正峰	   |0.995515823364258	        |針對政府準備修法推動「二代健保」，健保保費採取「年度所得總額」為計算基礎，而非採用扣除免稅額...|	National; Health Insurance; Insurance Premium  |Second-generation health insurance; total annual income|
| 彭添富	   |0.992780447006226	        |針對「辦理九十四年原住民中低收入戶家庭租屋補助計畫」專案補助計畫，特向行政院提出質詢。	    |Aboriginal life|	Aboriginal low- and middle-income households; housing subsidies|
| 李復興	   |0.992780089378357	        |發現自九十三年一月間起，勞保局陸續清查有一千多名國、公營事業退休員工溢領敬老津貼，截至93年...	|Old-age benefits; labor retirement|	Retired employees of public enterprises; over-receiving allowance for the elderly|
| 盧秀燕	   |0.992639720439911	        |針對早期退除役軍官給與補助金發放金額過低，實無法解決終身生活所需，希望相關單位考量實際情況，...	|Veterans welfare|	Grants for early retired officers|
| 李顯榮	   |0.990033149719238	        |對於陳水扁總統的「凱子外交」政策，不僅僅沒達到外交目的，更是浪費公帑，政府前後援賽金額高達5...	|Foreign aid; farmer welfare|	Triumphant diplomacy; old farmer allowance; Senegal|
| 丁守中	   |0.988385319709778	        |針就民眾陳情指出，目前政府對身心障礙者提供之生活津貼，依身心障礙程度等級分為1000元至50...	|Welfare for the handicapped|	Living allowance for the physically and mentally handicapped|
| 馮定國	   |0.985531985759735	        |鑒於國內經濟結構的快速調整，與人口高齡化的進展，未來中高齡失業問題必將日益嚴重，致使國人老年...	|Elderly welfare|	Aging; middle and old age unemployment|
| 彭添富	   |0.983698368072510	        |針對「豪雨成災，農作物損失補償」問題，特向行政院提出質詢。	|Agricultural subsidies |	Heavy rain; crops|
| 曾華德	   |0.979519009590149	        |為民國38年至43年間戌守大陳島等地區之中華民國前江、浙、閩、粵反共救國軍補發薪餉問題，攸關...	|Military pay|	Anti-Communist Salvation Army Reimbursement of Salary|
| 林鴻池	   |0.978044390678406	        |針對諸多已獲得五年五百億與卓越計畫獎補助的大學擬調漲學雜費，但教育部長杜正勝曾承諾，獲得上述...	|Education grant; education expenses; university |	Project Excellence Award subsidy; university tuition|
| 王昱婷	   |0.976034879684448	        |針對根據內政部最新統計，國內的嬰兒出生率再創新低點，今年1到4月只有6萬5400個小嬰兒出生...	|Fertility rate; population policy|	Birth rate; fertility rate|
| 彭添富	   |0.974732995033264	        |針對「觀音鄉保生社區風貌營造規劃設計」專案補助計畫，特向行政院提出質詢。	|Community project; government subsidy|	Baosheng community|
| 彭紹瑾	   |0.970751523971558	        |針對政府為提高生育率，有意將「育嬰假」放寬至全體勞工，並增加六個月的「育嬰留職停薪津貼」，此...	|Women's Welfare|	Parental leave; leave without pay allowance|
| 吳志揚	   |0.968230128288269	        |針對政府打著照顧中產階級的漂亮旗號，擬調增受薪大眾的薪資特別扣除額，但是只在薪資扣除額調整幅...	|Salary deduction|	Special salary deduction|








&nbsp; 

#### Last 10 Rows of 2000 Samples (less likely to pork barrel)


| Legislator | Pork/Constituency Interest |                               Legislative Questions                          |       Topic      |    Key Word    |
|:----------:|:--------------------------:|:----------------------------------------------------------------------------:|:-----------------|:--------------:|
|李復甸	   |0.000021549063604	|鑑於刑事偵察實務上緩起訴制度，有淪於檢察官為同案被告間不利證詞取得之交換手段之虞，破壞緩起訴...	|Investigation; litigation procedure |	Criminal investigation; secret witness|
|林建榮	   |0.000020212990421   |為立法院朝野協商修改銀行法，明定信用卡、現金卡循環利率與銀行公布的基本放款利率差距不得超過十...	|Financial management; bank management|	Banking Law; Cash Card; Revolving Interest Rate; Card Debt|
|林正峰	   |0.000019731034627	|針對行政院長張俊雄日前穿著輕便的長袖白襯衫，要求各級機關和學校身體力行節約能源，當場台下官員...	|Energy policy|	Energy saving|
|林正峰	   |0.000019187420548	|鑑於近年來臺灣地區毒品氾濫，吸毒人數劇增，危害國民身心健康甚鉅，因而滋生之犯罪更成為影響社會...	|Tobacco Restriction; Hospital|	Drug Abuse; Departmental Hospital; Special Agency for Drug Rehabilitation|
|王幸男	   |0.000017634354663	|針對道路人孔蓋或管線挖掘後回填品質不佳，或是公共安全沒有做好，導致各種傷害和死亡案件，一直居...	|Public safety|	Manhole cover; public safety; road quality|
|管碧玲	   |0.000013002485503	|針對近日台灣鐵路管理局發生網路訂票系統遭到內部人員惡意壟斷，導致一般民眾訂票權益受損之弊端；...	|Railway management; ticket	|Online booking; monopoly; Taiwan Railway|
|黃敏惠	   |0.000011985112906	|就近日來爆發知名提神飲料遭下毒事件，已知有四位台中市民因誤喝中毒，並有一人已不治死亡。此一類...	|Drinks; Poisoning|	Drinks; Poisoning|
|陳朝龍	   |0.000011277100384	|針對英國政府宣稱台灣出口至該國禽鳥，檢驗出感染禽流感H5N1病毒死亡。由於我國迄今並未發現有...	|Infectious disease prevention and control; smuggling|	British Government; Taiwanese birds; Avian Influenza; Smuggling|
|林進興	   |0.000007685628589	|針對行政院金融監督管理委員會為了解決國人廣大的卡債問題，積極推廣債務協商機制，政策本意良好，...	|Credit card; debt|	Card debt; negotiation mechanism|
|賴清德	   |0.000006586606105	|針對市售豆類製品疑含「過氧化氫」情形嚴重，傷害消費者健康，爰要求相關單位依食品衛生管理法切實...	|Food Management |Soy Products; Hydrogen Peroxide|
|邱毅	        |0.000006585601113	 |針對新聞局認定TVBS應為綜合台非新聞台乙案，準備將TVBS轉頻一事，日前行政院新聞局認定T...	|Freedom of the press; TV station|	Ownership Structure; Foreign Investment; Organic Law of the National Communications Commission; Freedom of Reporting|
|羅世雄	   |0.000006475097052	|針對手機通訊業者辦理新辦戶及更換SIM卡程序中，出現犯罪集團持偽造身分證，藉此竊取個人資料，...	|Telecommunications administration; national identity card; privacy; forgery|Mobile phone; personal data theft; criminal group|
|周守訓	   |0.000006256129382	|針對日前媒體報導陳水扁總統宣示，十二月二十五日耶誕節是重要的宗教節慶，更是我國的行憲紀念日，...	|Religion; Holiday|Christmas; Constitutional Anniversary; Religious Freedom; Separation of Church and State|
|郭榮宗	   |0.000004869816621	|對於板橋地方法院安全不設防？煙毒犯在地檢署廁所施打毒品，死亡兩天才被人發現，凸顯法警素質不足...	|District Court; Drugs|Banqiao District Court; drug offenders; drug abuse|
|潘孟安	   |0.000002590457370	|就立法委員選舉，改採單一選區兩票制即將首度實施，中央選舉委員應加強宣導「單一選區兩票制」的新...|	election|Legislative elections; two-vote system for a single constituency|





### Use End-to-End Model

```
from tensorflow import keras
model = keras.models.load_model('lour_pork_model') 
```

Step-by-step tutorial finds [here](https://github.com/davidycliao/PorkCNN/blob/main/demo-cnn-pork-barrel-classification-task.ipynb)

## Reference:

- [Yoon Kim, Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
- @gaussic's repo [text-classification-cnn-rnn](https://github.com/gaussic/text-classification-cnn-rnn)
- The collection of legislation was manually labelled by  Profession Luor, Ching-Jyuhn  and his research team.  I appreciate the assistance in providing the dataset.
- Chapter 11, 13, 14 from [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
