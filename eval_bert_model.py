import torch
import numpy as np
from transformers import BertTokenizer, BertPreTrainedModel, BertModel
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW,SGD,Adam
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch import nn
from sklearn.metrics import accuracy_score
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import re
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
#資料前處理
def stop(data):
    punctuation =  ['~', '`', '``', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', '-', '=', '{', '}', '|', '[',']', '\\', ':', '\"', ';', '\'', '<', '>', '?', ',', '.', '/','...','..','’',"''"]
    token =word_tokenize(data)
    tokenpun = [word for word in token if word not in punctuation]
    stops = set(stopwords.words('english'))
    tokenstop = [word for word in tokenpun if word not in stops]
    return tokenstop
#模型建立
class BertMultiLabelClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertMultiLabelClassifier, self).__init__(config)
        self.bert = BertModel(config)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.pooler_output)
        attention_weight=outputs[-1]
        return logits,attention_weight

tokenizer = BertTokenizer.from_pretrained('/home/d851211/bert-base-cased_tokenizer/1015最高閥值0.3用v1')#載入tokenizer
model = BertMultiLabelClassifier.from_pretrained('/home/d851211/bert-base-cased_model/1015最高閥值0.3用v1', num_labels=28,output_attentions=True)#載入model和確認結果數量
all_labels=[]
predicted_class_labels=[]
title=['admiration','amusement','anger','annoyance','approval','caring','confusion','curiosity','desire','disappointment','disapproval','disgust','embarrassment','excitement','fear','gratitude','grief','joy','love','nervousness','optimism','pride','realization','relief','remorse','sadness','surprise','neutral']
#匯入要預測的資料
df=pd.read_csv('/home/d851211/goemotions_validation.csv')
df1=pd.DataFrame()
df2=pd.DataFrame()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)

#model預測
model.eval()
for i in range(0,len(df)):
    onelabel=[]
    for j in range(0,len(title)):
        onelabel.append(int(df[title[j]][i]))
    new_text = ' '.join(stop(df['text'][i]))#stop(df['text'][i])
    all_labels.append(onelabel)
    encoding = tokenizer.encode_plus(
        new_text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    with torch.no_grad():
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs,attention_weight = model(input_ids, attention_mask=attention_mask)

    predicted_class = torch.sigmoid(outputs)
    predicted_class_list=predicted_class.cpu().numpy().tolist()[0]
    predicted_dict={
            'admiration':predicted_class_list[0],
            'amusement':predicted_class_list[1],
            'anger':predicted_class_list[2],
            'annoyance':predicted_class_list[3],
            'approval':predicted_class_list[4],
            'caring':predicted_class_list[5],
            'confusion':predicted_class_list[6],
            'curiosity':predicted_class_list[7],
            'desire':predicted_class_list[8],
            'disappointment':predicted_class_list[9],
            'disapproval':predicted_class_list[10],
            'disgust':predicted_class_list[11],
            'embarrassment':predicted_class_list[12],
            'excitement':predicted_class_list[13],
            'fear':predicted_class_list[14],
            'gratitude':predicted_class_list[15],
            'grief':predicted_class_list[16],
            'joy':predicted_class_list[17],
            'love':predicted_class_list[18],
            'nervousness':predicted_class_list[19],
            'optimism':predicted_class_list[20],
            'pride':predicted_class_list[21],
            'realization':predicted_class_list[22],
            'relief':predicted_class_list[23],
            'remorse':predicted_class_list[24],
            'sadness':predicted_class_list[25],
            'surprise':predicted_class_list[26],
            'neutral':predicted_class_list[27],
        }
    predicted_df=pd.DataFrame([predicted_dict])
    df1=pd.concat([df1,predicted_df],ignore_index=True)
    head_attention=attention_weight[-1][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    token_weights = [0] * len(tokens)
    #print(attention_weight)
    for head in head_attention:
        for i in range(len(tokens)):
            token_weights[i] += float(head[i][0])
    #for i in range(len(tokens)):
    attention_dict={
            'word':tokens,
            'attention':token_weights
        }
    attention_df=pd.DataFrame(attention_dict)
    df2=pd.concat([df2,attention_df],ignore_index=True)
        #print(tokens[i], token_weights[i])
    #print(tokens)
#結果匯出
df1.to_csv('test.csv')
df2.to_csv('test2.csv')