# PorkCNN 🐖🐖🐖
A Small Project for Pork Barrel Legislation  Classification Using CNN 


## Enviroment Setting

- Python 3.8 
- tensorflow==2.5.0
- numpy
- scikit-learn
- scipy 



### Num of Train/ Test 

```
Shape of X Train: (797, 123) 
Shape of X Test:  (394, 123) 
Shape of Y Trian: (797,) 
Shape of Y Test:  (394,)

# need more data to train 😥

```


### Model Evaluation

```
                precision    recall  f1-score   support

        Pork       0.92      0.95      0.94       265
    Not Pork       0.90      0.82      0.86       129

    accuracy                           0.91       394
   macro avg       0.91      0.89      0.90       394
weighted avg       0.91      0.91      0.91       394
```

### Application on New Dataset




| Pork Value(Constituency Interest) |      Legislative Questions   |  Topic |  Key Word |
|:----------:|:-------------:|:------:|:------:|
| 0.851461708545685 | 針對諸多已獲得五年五百億與卓越計畫獎補助的大學擬調漲學雜費，但教育部長杜正勝曾承諾，獲得上述...| 教育補助 ; 教育費用 ; 大學 | 卓越計畫獎補助 ; 大學學費       |
| 0.775758385658264 | 南投縣為農業縣份，工商不發達，稅收財源不豐裕，全年度自有財源收入尚不敷支應全縣人事費支出，而...| 地方財政 | 稅收 ;地方財政      |
| 0.716268658638000 | 針對內政部公告「低收入戶之資格家庭總收入以外財產總額之一定限額」有關不動產（土地及房屋）限額...| 低收入戶 ; 不動產 ; 政府補助 | 低收入戶 ; 不動產 ; 低收入戶補助 |
| 0.562548756599426 | 針對政府公教貸款利率高於一般商業銀行專案辦理之房屋貸款利率，使公教人員之房貸利息負擔沈重依舊...| 公務員福利 ; 房屋貸款 ; 利率| 公教貸款利率      |
| 0.337412178516388 | 針對台北縣板橋市大漢溪沿岸的環河道路，因道路寬窄不一，規劃不完善且無汽、機車分流，加上大型車...| 交通安全 ; 道路工程 | 大漢溪 ; 環河道路     |
| 0.201411515474319 | 針對目前政府十三萬臨時人員已依勞退新制提撥退休金，但仍尚未納入勞基法保障，政府應儘速公布將臨...| 勞動基準 ; 聘僱人員 ; 法律適用範圍| 臨時人員 ; 勞基法     |
| 0.148840516805649 | 針對內政部即將公布之「人口政策白皮書」，擬加碼發放育兒津貼，以獎勵生育。本席樂觀其成，惟希望...| 人口政策 | 人口政策     |
| 0.104532867670059 | 針對近年來財務持續惡化的中央健康保險局，利用健保費給付銀行巨額借款利息表示斥責。健保局用民眾...| 國民 ; 健康保險 ; 保險費 | 財務 ; 中央健康保險局 ; 健保費給付    |





### Use End-to-End Model

```
from tensorflow import keras
model = keras.models.load_model('lour_pork_model') 

```

Step-by-step tutorial finds [here](https://github.com/davidycliao/PorkCNN/blob/main/demo-cnn-pork-barrel-classification-task.ipynb)

## Reference:

- [Yoon Kim, Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
- @gaussic's repo [text-classification-cnn-rnn](https://github.com/gaussic/text-classification-cnn-rnn)
