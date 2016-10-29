import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
def read_data():

    #df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
    #ref detail : https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.names


    #read from csv
    filepath=os.path.dirname(os.path.realpath(__file__))#root, where the apk_integration_test.py file is.
    source_folder='source'
    data_csv_path=filepath+'/'+source_folder+'/breast_cancer.csv'
    df = pd.read_csv(data_csv_path, header=None)


    #save csv
    '''
    filepath=os.path.dirname(os.path.realpath(__file__))#root, where the apk_integration_test.py file is.
    source_folder='source'
    data_csv_path=filepath+'/'+source_folder+'/breast_cancer.csv'
    df.to_csv(data_csv_path,index=0,header=False)
    '''

    #print('Class labels', np.unique(df['Class label']))
    print(df.head())
    print(df.shape)
    return df




def main():
    df=read_data()

    ################
    # data preprocessing
    ################
    from sklearn.preprocessing import LabelEncoder
    #assign 30 features set to X.
    X = df.loc[:, 2:].values
    #assign [1] to label y
    y = df.loc[:, 1].values
    #print(y)

    #transform y label M,B to 1,0
    le = LabelEncoder()
    y = le.fit_transform(y)
    le.transform(['M', 'B'])
    #print(y)

    ############
    # separate train, test set
    #############
    from sklearn.cross_validation import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)


    print("Data ready")

    ###################
    # setup pipeline
    #####################
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    pipe_lr = Pipeline([('scl', StandardScaler()),
            ('clf', LogisticRegression(penalty='l2', random_state=0))])


    ################
    # do the simple cross_validation index seletion
    ###################
    # we only take 2 features(column4 and 14 ) to X_train2, that is because to demo only.
    # the resason the same as below.
    X_train2 = X_train[:, [4, 14]]
    #X_train2=X_train2
    #print(X_train)
    #print(X_train2)

    from sklearn.cross_validation import StratifiedKFold
    #we chose folds =3 is just for create the bad rsult of classifer,
    # because we want to see the different in ROC pic
    # usually n_foldes=10
    cv = StratifiedKFold(y_train, n_folds=3, random_state=1)


    #plot the roc curve
    from sklearn.metrics import roc_curve, auc
    from scipy import interp


    fig = plt.figure(figsize=(7, 5))

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i, (train, test) in enumerate(cv):
        probas = pipe_lr.fit(X_train2[train],
                             y_train[train]).predict_proba(X_train2[test])

        fpr, tpr, thresholds = roc_curve(y_train[test],
                                         probas[:, 1],
                                         pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr,
                 tpr,
                 lw=1,
                 label='ROC fold %d (AUC area = %0.2f)'
                        % (i+1, roc_auc))



    plt.plot([0, 1],
             [0, 1],
             linestyle='--',
             color=(0.6, 0.6, 0.6),
             label='random guessing')

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.plot([0, 0, 1],
             [0, 1, 1],
             lw=2,
             linestyle=':',
             color='black',
             label='perfect performance')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('Receiver Operator Characteristic')
    plt.legend(loc="lower right")

    plt.tight_layout()
    # plt.savefig('./figures/roc.png', dpi=300)
    plt.show()



    #####################
    # if we only want to see  ROC AUC of any model
    #####################
    from sklearn.svm import SVC #SVM model
    pipe_svc = Pipeline([('scl', StandardScaler()),
                ('clf', SVC(random_state=1))])

    pipe_svc = pipe_svc.fit(X_train2, y_train)
    y_pred2 = pipe_svc.predict(X_test[:, [4, 14]])
    from sklearn.metrics import roc_auc_score, accuracy_score
    print('ROC AUC: %.3f' % roc_auc_score(y_true=y_test, y_score=y_pred2))
    print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred2))
    '''
    ROC AUC: 0.671
    Accuracy: 0.728

    '''

    '''
    #########################
    # Set the Scoring metrics for multiclass classification with precision_score
    ##########################

    pre_scorer = make_scorer(score_func=precision_score,
                         pos_label=1,
                         greater_is_better=True,
                         average='micro')
    '''
    return 0

main()
