from nltk.tokenize import sent_tokenize, word_tokenize 
import gensim
import os
import collections
import smart_open
import random
import json
import pandas as pd
import numpy as np
n = 10


test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
lee_test_file = test_data_dir + os.sep + 'lee.cor'

def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])


train_corpus = list(read_corpus(lee_train_file))
test_corpus = list(read_corpus(lee_test_file, tokens_only=True))

model = gensim.models.doc2vec.Doc2Vec(vector_size=n, min_count=2, epochs=40)
model.build_vocab(train_corpus)




xls = pd.ExcelFile('testData.xlsx')
df1 = pd.read_excel(xls, 'Sheet1')

sLength = len(df1['title'])

for i in range(0,n):
        name = 'doc2vec'+str(i)
        df1[name] = pd.Series(np.zeros(sLength), index=df1.index)


count = 0
failed = 0


for index, row in df1.iterrows():
        title = str(row['title'])
        year = str(row['yearWritten'])
        
        try:
                with open('../books/'+year+'/'+title+'/review.txt') as f:
                        s1 = f.read() 
                        s2 = s1.replace("\n", " ") 
                        data = [] 

                        for i in sent_tokenize(s2): 
                            for j in word_tokenize(i): 
                                data.append(j.lower())

                        docVector = model.infer_vector(data)
                        for i in range(0,n):
                            name = 'doc2vec'+str(i)
                            df1.loc[df1.index[index], name] = docVector[i]


                        # for i in range(0,11):
                        #         name = 'vggClass'+str(i)
                        #         data[name] = data[name] * weigthCorection[i]
                        #         sum = sum + data[name]

                        # for i in range(0,11):
                        #         name = 'vggClass'+str(i)
                        #         df1.loc[df1.index[index], name] = round(data[name] / sum, 4)
                count += 1
                print('count',count)
        except:
                failed += 1
                print('failed ',failed)

writer = pd.ExcelWriter('testData'+str(n)+'.xlsx')
df1.to_excel(writer)
writer.save()






# sample = open("review.txt", "r") 
# s = sample.read() 
  
# # Replaces escape character with space 
# f = s.replace("\n", " ") 
  
# data = [] 

 
# for i in sent_tokenize(f): 
    
#     for j in word_tokenize(i): 
#         data.append(j.lower()) 


# print(data)
# print(model.infer_vector(data))








