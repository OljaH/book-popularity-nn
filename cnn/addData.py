import os
import json
import pandas as pd
import numpy as np

# books = dict()
# with open('final_fajl_5.csv','r') as f:
#         for l in f:
#                 if(l != '\n'):
#                         linija = l.strip()
#                         split = linija.split(',')
#                         books[split[0]] = split[1]


weigthCorection = [47.6,17.54,8.13,24.39,22.22,8.47,30.3,1.38,17.24,27.77,20.83]

xls = pd.ExcelFile('train_data (1).xlsx')
df1 = pd.read_excel(xls, 'Sheet1')

sLength = len(df1['title'])

for i in range(0,11):
        name = 'vggClass'+str(i)
        df1[name] = pd.Series(np.zeros(sLength), index=df1.index)


count = 0
failed = 0

for index, row in df1.iterrows():
        title = str(row['title'])
        year = str(row['yearWritten'])[:-2]
        
        try:
                with open('../books/'+year+'/'+title+'/vgg16Classification.json') as f:
                        data = json.load(f)
                        sum = 0
                        for i in range(0,11):
                                name = 'vggClass'+str(i)
                                data[name] = data[name] * weigthCorection[i]
                                sum = sum + data[name]

                        for i in range(0,11):
                                name = 'vggClass'+str(i)
                                df1.loc[df1.index[index], name] = round(data[name] / sum, 4)
                count += 1
                print('count',count)
        except:
                failed += 1
                print('failed ',failed)

writer = pd.ExcelWriter('trainData.xlsx')
df1.to_excel(writer)
writer.save()

