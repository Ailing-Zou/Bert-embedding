# Bert-embedding

## The purpose:
To extract bert embedding from text, we just want to get the embedding by bert.

## The background:
Bert as service is not suitable for my work requirement.

## Where I collect these methods:
from github list and google, the most related ones.

## Method collection 
( including some out of date method to give a whole description ) :
### 1) Bert-embedding: 

https://github.com/imgarylai/bert-embedding you can use it by: 

    pip install bert-embedding

[It will probably the first one appearing in the related list. However ,it is deprecated because the author has no time to maintain it, its output shows some unreasonable points.]

eg. an issue in bert-embedding

    When using 
    bert_embedding = BertEmbedding(model='bert_12_768_12', dataset_name='wiki_cn')
    res = bert_embedding(u'火箭险胜勇士')
    There comes 2 [UNK] for each Chinese character, like this:
    [(['[UNK]', '火', '[UNK]'], ...)]
    and it is followed by 3 768-dimension arrays.
 
    Which one is exactly the word embedding for "火"?  
    The author doesn't give answer for this.

### 2) 
   https://github.com/weiarqq/bert-embedding 
   - Implemented by tensorflow
   - Suitable for industrial use  
   - Easy to use

### 3) 
   https://github.com/YC-wind
   - Based on bert as service
   - Implemented by tensorflow
   
### 4) 
   https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
   - This is one tutorial for getting embedding from Chris McCormick and Nick Ryan.
   - Very simple and easy to scan.
   - The original version is jupyternook, I do some minor changes for my chinese usage, maybe you can download bert-word-embedding.py in this repository directly.

