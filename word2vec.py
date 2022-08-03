import pandas as pd
import numpy as np
import csv
import re
import nltk
from sklearn.ensemble import GradientBoostingRegressor
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim
from gensim.models import KeyedVectors, Doc2Vec, Word2Vec
from gensim.models.doc2vec import TaggedDocument
import warnings

warnings.filterwarnings('ignore')


class MyTokenizer:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_X = []
        for document in X:
            tokenized_doc = []
            for sent in nltk.sent_tokenize(document):
                tokenized_doc += nltk.word_tokenize(sent)
            transformed_X.append(np.array(tokenized_doc))
        return np.array(transformed_X)

    def fit_transform(self, X, y=None):
        return self.transform(X)


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.wv.vectors[0])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = MyTokenizer().fit_transform(X)

        return np.array([
            np.mean([self.word2vec.wv[w] for w in words if w in self.word2vec.wv]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

    def fit_transform(self, X, y=None):
        return self.transform(X)


data = pd.read_csv(file).drop([columns_name],axis=1)

# drop nan
df = data.dropna(subset=[columns_name])

# normalize
# this step is to decrease range in some columns that have large values, for example
df['WINDOWHEIGHT_NORMALIZE'] = (df['WINDOW_HEIGHT'] - df['WINDOW_HEIGHT'].mean()) / df['WINDOW_HEIGHT'].std()


# choose categorical columns that we want to convert to vectors
df1 = df[['COUNTRY_GROUP', 'BROWSER_GROUP','UNIT_SUBTYPE', 'APP_SESSION_ID','PAGE_URL', 'BROWSER_NAME']]
df2 = df1.apply(lambda x: ','.join(x.astype(str)), axis=1)
df_clean = pd.DataFrame({'clean': df2})
df_merge = pd.concat([df, df_clean.reindex(df1.index)], axis=1)
clean_txt = []

for w in range(df_merge.shape[0]):
    desc = df_merge['clean'].iloc[w].lower()
    # remove punctuation
    desc = re.sub('[^a-zA-Z]', ' ', desc)
    # remove tags
    desc = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", desc)
    # remove digits and special chars
    desc = re.sub("(\\d|\\W)+", " ", desc)
    clean_txt.append(desc)
df_merge['clean_final'] = clean_txt
corpus = []
for col in df_merge.clean_final:
    word_list = col.split(" ")
    corpus.append(word_list)

# generate vectors from corpus
ssize = 16  # ssize is the vector size we want
model = Word2Vec(corpus, min_count=1, vector_size=ssize)

mean_embedding_vectorizer = MeanEmbeddingVectorizer(model)
mean_embedded = mean_embedding_vectorizer.fit_transform(df_merge['clean_final'])
df_merge['ARRAY'] = list(mean_embedded)
df_final = df_merge.drop(['COUNTRY_GROUP', 'BROWSER_GROUP','UNIT_SUBTYPE', 'APP_SESSION_ID',
                          'PAGE_URL', 'BROWSER_NAME','clean', 'clean_final'], axis=1)

df_final[[f'{i}' for i in range(ssize)]] = pd.DataFrame(df_final.ARRAY.values.tolist(), index=df_final.index)
df_final = df_final.drop(['ARRAY'], axis=1)
ppp = df_final.pop('CPM')
df_final.insert(df_final.shape[1],'CPM',ppp)
# use df_final as the final dataset for training data





