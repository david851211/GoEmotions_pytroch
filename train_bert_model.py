import torch
import numpy as np
from transformers import BertTokenizer, BertPreTrainedModel, BertModel
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW,SGD,Adam
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch import nn
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import re
#用於建立資料集
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):#
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)#self.max_len

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        if len(label) >28:
            print(len(label))
            print(label)
            print(idx)
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label':  torch.tensor(label, dtype=torch.float) 
        }
#用於模型建立
class BertMultiLabelClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertMultiLabelClassifier, self).__init__(config)
        self.bert = BertModel(config)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.pooler_output)
        return logits

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')#載入tokenizer
model = BertMultiLabelClassifier.from_pretrained('bert-base-cased', num_labels=28)#載入模型和確認輸出結果
df=pd.read_csv('/home/d851211/goemotions_train.csv')#載入訓練集
df1=pd.read_csv('/home/d851211/goemotions_test.csv')#載入測試集
title=['admiration','amusement','anger','annoyance','approval','caring','confusion','curiosity','desire','disappointment','disapproval','disgust','embarrassment','excitement','fear','gratitude','grief','joy','love','nervousness','optimism','pride','realization','relief','remorse','sadness','surprise','neutral']
#用於處理匯入資料
texts_train=[]
labels_train=[]
texts_val=[]
labels_val=[]
for doc in range(0,len(df)):
    texts_train.append(df['text'][doc])
    onelabel=[]
    for j in range(0,len(title)):
        onelabel.append(int(df[title[j]][doc]))
    labels_train.append(onelabel)
for doc in range(0,len(df1)):
    onelabel1=[]
    texts_val.append(df1['text'][doc])
    for j in range(0,len(title)):
        onelabel1.append(int(df1[title[j]][doc]))
    labels_val.append(onelabel1)
#用於建立dataloader
train_dataset = TextClassificationDataset(texts_train, labels_train, tokenizer, max_len=128)
train_data_loader = DataLoader(train_dataset , batch_size=128, shuffle=True, num_workers=4)
val_dataset = TextClassificationDataset(texts_val, labels_val, tokenizer, max_len=128)
val_data_loader = DataLoader(val_dataset , batch_size=128, shuffle=True, num_workers=4)
#確認GPU和使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)#確認優化器
epochs = 10  # 決定訓練次數
criterion = nn.BCEWithLogitsLoss()#決定loss
best_loss = float('inf')
count = 0
for epoch in range(epochs):
    
    train_loss = 0
    val_loss = 0
    all_labels=[]
    predicted_class_labels=[]
    #model訓練
    model.train()
    for batch in train_data_loader :
                
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)      
        outputs = model(input_ids, attention_mask=attention_mask)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()        
        optimizer.step()
        train_loss += loss.item() * input_ids.size(0)
    train_loss = train_loss / len(train_data_loader.dataset)
    print(f"Epoch {epoch+1} / {epochs}, Training Loss: {train_loss}")
    #model評估
    model.eval()
    with torch.no_grad():
        for batch in val_data_loader :
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            predicted_class = torch.sigmoid(outputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * input_ids.size(0)
    val_loss = val_loss / len(val_data_loader.dataset)
    print(f"Epoch {epoch+1} / {epochs}, val Loss: {val_loss}")
    #tokenizer和model匯出
    tokenizer.save_pretrained(f'/home/d851211/bert-base-cased_tokenizer/{epoch}/')
    model.save_pretrained(f'/home/d851211/bert-base-cased_model/{epoch}/')
    #判斷訓練結果
    print(f'best_loss:{best_loss}')
    if val_loss < best_loss:
        best_loss = val_loss
        count = 0
    else :
        count +=1
    if count >=10:
        break
    


