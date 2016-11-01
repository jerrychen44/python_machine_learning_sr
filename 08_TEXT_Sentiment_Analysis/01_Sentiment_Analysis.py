# -*- coding: utf-8 -*-

import pyprind
import pandas as pd
import os
import numpy as np
import nltk



filepath=os.path.dirname(os.path.realpath(__file__))#root, where the apk_integration_test.py file is.
source_folder='source'
#print(filepath)
data_csv_path=filepath+'/'+source_folder+'/movie_data.csv'
#print(data_csv_path)

def combine_datafile_to_csv():

    #download the original file http://ai.stanford.edu/~amaas/data/sentiment/
    labels = {'pos':1, 'neg':0}
    pbar = pyprind.ProgBar(50000)
    df = pd.DataFrame()
    for s in ('test', 'train'):
        for l in ('pos', 'neg'):
            path =filepath+'/'+source_folder+'/aclImdb/%s/%s' % (s, l)
            for file in os.listdir(path):
                with open(os.path.join(path, file), 'r',encoding='utf-8') as infile:
                    txt = infile.read()
                    #print(type(txt))
                df = df.append([[txt, labels[l]]], ignore_index=True)
                pbar.update()
    df.columns = ['review', 'sentiment']


    #Shuffling the DataFrame:
    np.random.seed(0)
    df = df.reindex(np.random.permutation(df.index))
    print(df.head())
    df.to_csv(data_csv_path, index=False)

    return 0

def read_csv_pd():
    #df = pd.read_csv('https://archive.ics.uci.edu/ml/'
    #        'machine-learning-databases/iris/iris.data', header=None)
    df = pd.read_csv(data_csv_path)

    #df.to_csv(filepath+'/'+source_folder+'/iris.csv',index=0,header=False)

    print(df.head())
    print(df.shape)
    return df

def bag_of_words_example():
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer

    ################
    # Transforming documents into feature vectors
    ################

    count = CountVectorizer()
    docs = np.array([
            'The sun is shining',
            'The weather is sweet',
            'The sun is shining and the weather is sweet'])
    bag = count.fit_transform(docs)

    print(count.vocabulary_)
    '''{'weather': 6, 'sweet': 4, 'shining': 2, 'and': 0, 'the': 5, 'is': 1, 'sun': 3}'''
    print(bag.toarray())
    '''
     [[0 1 1 1 0 1 0]
     [0 1 0 0 1 1 1]
     [1 2 1 1 1 2 1]]'''


    ################
    # Assessing word relevancy via term frequency-inverse document frequency
    ###############
    np.set_printoptions(precision=2)
    from sklearn.feature_extraction.text import TfidfTransformer

    tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
    print(tfidf.fit_transform(count.fit_transform(docs)).toarray())
    '''
    [[ 0.    0.43  0.56  0.56  0.    0.43  0.  ]
     [ 0.    0.43  0.    0.    0.56  0.43  0.56]
     [ 0.4   0.48  0.31  0.31  0.31  0.48  0.31]]
     '''

    # we try count by ourself
    tf_is = 2
    n_docs = 3
    idf_is = np.log((n_docs+1) / (3+1) )
    tfidf_is = tf_is * (idf_is + 1)
    print('tf-idf of term "is" = %.2f' % tfidf_is)
    '''
    tf-idf of term "is" = 2.00
    '''
    tfidf = TfidfTransformer(use_idf=True, norm=None, smooth_idf=True)
    raw_tfidf = tfidf.fit_transform(count.fit_transform(docs)).toarray()[-1]
    print(raw_tfidf )
    '''
    [ 1.69  2.    1.29  1.29  1.29  2.    1.29]
    '''

    l2_tfidf = raw_tfidf / np.sqrt(np.sum(raw_tfidf**2))
    print(l2_tfidf)

    '''
    [ 0.4   0.48  0.31  0.31  0.31  0.48  0.31]
    '''

    return 0

import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + \
           ' '.join(emoticons).replace('-', '')
    return text

def cleaning_text_data(df):
    print(df.loc[0, 'review'][-50:])
    print(preprocessor(df.loc[0, 'review'][-50:]))
    df['review'] = df['review'].apply(preprocessor)

    return df

def tokenizer(text):
    return text.split()

from nltk.stem.porter import PorterStemmer
def tokenizer_porter(text):


    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]


def documents_into_tokens_example():


    print(tokenizer('runners like running and thus they run'))
    '''['runners', 'like', 'running', 'and', 'thus', 'they', 'run']'''
    print(tokenizer_porter('runners like running and thus they run'))
    '''['runner', 'like', 'run', 'and', 'thu', 'they', 'run']'''

    return 0

from nltk.corpus import stopwords
def stop_word_remove():

    #updated
    nltk.download('stopwords')

    global stop
    stop = stopwords.words('english')
    result=[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop]
    print(result)
    '''['runner', 'like', 'run', 'run', 'lot']'''
    return 0


def example(df):
    ################
    # bag-of-words model
    ##################
    bag_of_words_example()


    ##############
    # Cleaning text data
    ###############
    cleaning_text_data(df)

    ################
    # Processing documents into tokens
    #################
    documents_into_tokens_example()



    ################
    # stop word remove
    #################
    stop_word_remove()
    return 0

def main():


    #conver original file to csv for
    #futuer use.
    #combine_datafile_to_csv()

    df=read_csv_pd()

    #try example
    #example(df)

    ##############
    # Cleaning text data
    ###############
    df=cleaning_text_data(df)
    #init stop word
    stop_word_remove()

    X_train = df.loc[:25000, 'review'].values
    y_train = df.loc[:25000, 'sentiment'].values
    X_test = df.loc[25000:, 'review'].values
    y_test = df.loc[25000:, 'sentiment'].values

    from sklearn.grid_search import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    # TfidfVectorizer() = CountVectorizer() + TfidfTransformer()
    tfidf = TfidfVectorizer(strip_accents=None,
                            lowercase=False,
                            preprocessor=None)

    param_grid = [{'vect__ngram_range': [(1,1)],
                   'vect__stop_words': [stop, None],
                   'vect__tokenizer': [tokenizer, tokenizer_porter],
                   'clf__penalty': ['l1', 'l2'],
                   'clf__C': [1.0, 10.0, 100.0]},
                 {'vect__ngram_range': [(1,1)],
                   'vect__stop_words': [stop, None],
                   'vect__tokenizer': [tokenizer, tokenizer_porter],
                   'vect__use_idf':[False],
                   'vect__norm':[None],
                   'clf__penalty': ['l1', 'l2'],
                   'clf__C': [1.0, 10.0, 100.0]},
                 ]

    lr_tfidf = Pipeline([('vect', tfidf),
                         ('clf', LogisticRegression(random_state=0))])

    gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                               scoring='accuracy',
                               cv=5, verbose=1,
                               n_jobs=-1)


    #fit data
    gs_lr_tfidf.fit(X_train, y_train)
    print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
    print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
    clf = gs_lr_tfidf.best_estimator_
    print('Test Accuracy: %.3f' % clf.score(X_test, y_test))
    '''
    #take 27 minutes
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:  3.0min
    [Parallel(n_jobs=-1)]: Done 192 tasks      | elapsed: 17.1min
    [Parallel(n_jobs=-1)]: Done 240 out of 240 | elapsed: 22.9min finished
    Best parameter set: {'clf__C': 10.0, 'vect__ngram_range': (1, 1), 'clf__penalty': 'l2', 'vect__tokenizer': <function tokenizer at 0x7f94eb1dc6a8>, 'vect__stop_words': None}
    CV Accuracy: 0.897
    Test Accuracy: 0.898
    '''
    return 0
main()
