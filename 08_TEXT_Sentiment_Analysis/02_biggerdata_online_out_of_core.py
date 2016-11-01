import pyprind
import pandas as pd
import nltk
import numpy as np
import re,os
from nltk.corpus import stopwords

stop = stopwords.words('english')
filepath=os.path.dirname(os.path.realpath(__file__))#root, where the apk_integration_test.py file is.
source_folder='source'
#print(filepath)
data_csv_path=filepath+'/'+source_folder+'/movie_data.csv'



def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

def stream_docs(path):
    with open(path, 'r') as csv:
        next(csv) # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


def main():
    #try try
    print(next(stream_docs(path=data_csv_path)))

    #data processing
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import SGDClassifier

    vect = HashingVectorizer(decode_error='ignore',
                             n_features=2**21,
                             preprocessor=None,
                             tokenizer=tokenizer)

    clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
    doc_stream = stream_docs(path=data_csv_path)





    import pyprind
    pbar = pyprind.ProgBar(45)

    # do 45 times mini train, and each trains has pick up 1000 data
    classes = np.array([0, 1])
    for _ in range(45):
        #get 1000 data
        X_train, y_train = get_minibatch(doc_stream, size=1000)
        if not X_train:
            break
        # data processing
        X_train = vect.transform(X_train)
        # fit the model.
        clf.partial_fit(X_train, y_train, classes=classes)
        #just update the tool bar
        pbar.update()

    #ok, above we train a model by 45x1000 data.

    #we use the same process (which means get_minibatch)
    # but diriectly get 5000 data to do the testing.

    X_test, y_test = get_minibatch(doc_stream, size=5000)
    X_test = vect.transform(X_test)
    #count score
    print('Accuracy: %.3f' % clf.score(X_test, y_test))
    '''Accuracy: 0.867'''

    # then also use those data to improve our model
    clf = clf.partial_fit(X_test, y_test)


    return 0


main()
