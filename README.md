# GoEmotions_pytroch
## 資料來源
* 使用google的情感資料集[GoEmotions](<https://research.google/blog/goemotions-a-dataset-for-fine-grained-emotion-classification/>)和使用pytroch建立的bert情感分類器
## GoEmotions是什麼
* GoEmotions是由google建立的包含12種正面情緒，11種負面情緒和4種曖昧不明的情緒以及1種中性情緒的情緒資料集，跟舊有的情緒資料集相比在需要細微區分情緒的任務中可以表現的更好
## 模型
*goole雲端([下載](<https://drive.google.com/file/d/1_cLnsseP1HzKokYgfcUnq8gJP7aV_L6T/view?usp=sharing>))
## 使用模型
* 使用bert模型是採用bert-base-cased模型
## 結果評估
*F1 score 為0.59
## 使用方法
*將要預測的資料放在eval_bert_model.py裡面就可以結果預測
## 預測結果
*eval_bert_model.py會給予每篇文本28種情感的預測機率和每個詞彙的attention層的結果可以用來判斷哪個詞彙在模型中是顯著的。
