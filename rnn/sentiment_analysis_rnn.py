import numpy as np
import pandas as pd
from keras.models import model_from_json, Sequential
from keras.layers import Dense, LSTM, GRU, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# training data set - from https://www.kaggle.com/oumaimahourrane/imdb-reviews
data = pd.read_csv("data/imdb_reviews_train.csv",  encoding = "ISO-8859-1")
print(data.shape)
X = data['SentimentText']
y = data['Sentiment']

def vectorize(list_of_words, vectorizer=None):
    if vectorizer==None:
        vectorizer = CountVectorizer()
        vectorizer.fit(list_of_words)
    transformed_data = []
    for row in list_of_words:
        embedding = []
        for n in row.split():
            if not (n.isdigit() or (n[0] == '-' and n[1:].isdigit())):
                embedding.append(vectorizer.vocabulary_.get(n, 0))
        transformed_data.append(embedding)
    transformed_data = pad_sequences(transformed_data, maxlen=100)
    print('VOCABULARY - {} elements: \n{}'.format(len(vectorizer.vocabulary_),
                                                  sorted(vectorizer.vocabulary_, key=lambda x: x[1])))
    return vectorizer, np.asarray(transformed_data)

Y = pd.get_dummies(data['Sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

vectorizer, xx_train = vectorize(X_train)
_,xx_test = vectorize(X_test, vectorizer)

t = Tokenizer()
t.fit_on_texts(X_train)
vocab_size = len(t.word_index) + 1
encoded_docs = t.texts_to_sequences(X_train)

embeddings_index = dict()
f = open('glove.6B.100d.txt', encoding="utf8")
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in t.word_index.items():
  embedding_vector = embeddings_index.get(word)
  if embedding_vector is not None:
    embedding_matrix[i] = embedding_vector

# create model
max_features = vocab_size
embed_dim = 100
lstm_out = 196
model = Sequential()
model.add(Embedding(max_features, embed_dim,input_length = xx_train.shape[1], weights=[embedding_matrix]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

# sentiments
yy_train = pd.get_dummies(Y_train)
yy_test = pd.get_dummies(Y_test)
print(xx_train.shape,yy_train.shape)
print(xx_test.shape,yy_test.shape)

# train the model
batch_size = 64
validation_size = 1500
model.fit(xx_train, yy_train, epochs = 10, batch_size=batch_size, verbose = 2)
xx_validate = xx_test[-validation_size:]
yy_validate = yy_test[-validation_size:]
xx_test = xx_test[:-validation_size]
yy_test = yy_test[:-validation_size]


# serialize the model and its weights
model_json = model.to_json()
with open("rnn_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('rnn_model_weights.h5')

json_file = open('rnn_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('rnn_model_weights.h5')
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])

score,acc = model.evaluate(xx_test, yy_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))

# model for testing
Test=False
if Test:

    testComments=[]
    lines = open("data/testComments/full.txt", encoding="utf8").read().splitlines()
    index=0
    test2=[]
    import langdetect as ld
    for l in lines:
        try:
            if ld.detect(str(l))=='en':
                test2.append(l)
                index = index + 1
        except:
            print("exception")

    f.close()
    _,xxxx_test = vectorize(test2, vectorizer)
    results = loaded_model.predict(xxxx_test)
    if index!=0:
        print(sum(results)/index)
    else:
        print("no data")

    yyyy_test = open("data/testComments/sent.txt", encoding="utf8").read().splitlines()
    print("*********")
    yyyy_test = pd.get_dummies(yyyy_test)
    score,acc = loaded_model.evaluate(xxxx_test, yyyy_test, verbose = 2, batch_size = batch_size)
    print("score: %.2f" % (score))
    print("acc: %.2f" % (acc))


    import glob
    # load train and test model for the third model
    model_train =  pd.read_excel('data/train_model.xlsx')
    url_train = list(model_train['url'])
    model_test=  pd.read_excel('data/test_model.xlsx')
    url_test = list(model_test['url'])
    model_train['sentiment']=-1.0
    model_test['sentiment']=-1.0

    for filepath in glob.iglob("data/bookCommentTextSummarized/*"):
        filepath=filepath.replace("\\", "/")
        f = open(filepath, encoding="utf8")
        lines = f.read().splitlines()
        index = 0
        test = []
        import langdetect as ld
        url_book = lines[0]
        for l in lines:
            try:
                if ld.detect(str(l))=='en':
                    test.append(l)
                    index = index + 1
            except:
                continue

        f.close()
        if test:
            _, xxxx_test = vectorize(test, vectorizer)
            results = loaded_model.predict_classes(xxxx_test)
            res = 0
            if index!=0:
                res=sum(results)/index
                print(res)
                try:
                    ind1 = url_train.index(url_book)
                    model_train.at[ind1, 'sentiment'] = res
                    print(model_train.at[ind1, 'sentiment'])
                except:
                    try:
                        ind2 = url_test.index(url_book)
                        model_test.at[ind2, 'sentiment'] = res
                    except:
                        continue
            else:
                print("no data")
        else:
            print("no english comments!")

    # save results to files for the third part of model
    print(model_train['sentiment'])
    model_train.to_excel("tr_m.xlsx")
    model_test.to_excel("te_m.xlsx")

    lines = open("data/bookCommentTextSummarized/test.txt", encoding="utf8").read().splitlines()
    index=0
    test2=[]
    import langdetect as ld
    for l in lines:

        # print(index)
        print(l)
        try:
            if ld.detect(str(l))=='en':
                test2.append(l)
                index = index + 1
                print(l)
        except:
            print("exception")


    f.close()
    _,xxxx_test = vectorize(test2, vectorizer)
    results = loaded_model.predict_classes(xxxx_test)
    print(results)
    if index!=0:
        print(sum(results)/index)
    else:
        print("no data")

    # test komentari 0.74 tacnost nad sum skracenim
    yyyy_test = open(filepath, encoding="utf8").read().splitlines()
    print("*********")
    yyyy_test = pd.get_dummies(yyyy_test)
    score,acc = loaded_model.evaluate(xxxx_test, yyyy_test, verbose = 2, batch_size = batch_size)
    print("score: %.2f" % (score))
    print("acc: %.2f" % (acc))