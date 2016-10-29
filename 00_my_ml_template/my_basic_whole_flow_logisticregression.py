import pandas as pd
import os,sys
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




def feature_preprocessing(X):
    from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler
    #stdsc = StandardScaler()
    stdsc = StandardScaler(with_mean=False)
    minmaxsc=MinMaxScaler(feature_range=(0, 1),copy=False)
    maxabssc=MaxAbsScaler()

    print(X[1])

    #X_train_std = stdsc.fit_transform(X_train)
    #X_test_std = stdsc.transform(X_test)


    #X_sc= stdsc.fit_transform(X)
    #X_train = stdsc.fit_transform(X_train)
    #X_test = stdsc.transform(X_test)




    X= minmaxsc.fit_transform(X)
    #X_train = minmaxsc.fit_transform(X_train)
    #X_test = minmaxsc.transform(X_test)

    #X= maxabssc.fit_transform(X)
    #X_train = maxabssc.fit_transform(X_train)
    #X_test = maxabssc.transform(X_test)



    #print(X_train[1])
    print(X[1])
    return X

def rbf_svc(X,y,X_train,y_train,X_test,y_test):
    from sklearn import svm
    rbf_svc = svm.SVC(kernel='rbf')
    rbf_svc=rbf_svc.fit(X_train, y_train)
    y_train_pred = rbf_svc.predict(X_train)
    y_test_pred = rbf_svc.predict(X_test)

    from sklearn.metrics import accuracy_score
    rbf_train = accuracy_score(y_train, y_train_pred)
    rbf_test = accuracy_score(y_test, y_test_pred)
    print('rbf_svc,  train/test accuracies %.3f/%.3f'
              % (rbf_train, rbf_test))

    ###########################
    # get the learning cruve
    ##########################
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, test_scores =\
                learning_curve(estimator=rbf_svc,
                X=X_train,
                y=y_train,
                train_sizes=np.linspace(0.1, 1.0, 10),
                cv=10
                )

    #plot it out
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    print("learning_curve train: %0.2f (+/- %0.2f) [rbf_svc]"
          % ( np.asarray(train_mean).mean(),  np.asarray(train_std).mean() ))

    print("learning_curve test: %0.2f (+/- %0.2f) [rbf_svc]"
          % ( np.asarray(test_mean).mean(),  np.asarray(test_std).mean() ))

    plt.plot(train_sizes, train_mean,
             color='blue', marker='o',
             markersize=5, label='training accuracy')

    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy')

    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.5, 1.0])
    #plt.tight_layout()
    # plt.savefig('./figures/learning_curve.png', dpi=300)
    plt.show()
    return 0


def logisticRegression(X,y,X_train,y_train,X_test,y_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    lr=LogisticRegression(random_state=0)
    lr=lr.fit(X_train, y_train)
    y_train_pred = lr.predict(X_train)
    y_test_pred = lr.predict(X_test)

    clf_train = accuracy_score(y_train, y_train_pred)
    clf_test = accuracy_score(y_test, y_test_pred)
    print('LogisticRegression,  train/test accuracies %.3f/%.3f'
              % (clf_train, clf_test))



        #print(lr.coef_,lr.coef_.shape)

    y_test_pred_prob=lr.predict_proba(X_test)
    print(type(y_test_pred_prob),y_test_pred_prob)
    print(type(y_test_pred),y_test_pred)



    #Evaluaton 1 : confusion matrix¶
    from sklearn.metrics import confusion_matrix
    confmat = confusion_matrix(y_true=y_test, y_pred=y_test_pred)
    print(confmat)

    #plot out
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

    plt.xlabel('predicted label')
    plt.ylabel('true label')

    #plt.tight_layout()
    # plt.savefig('./figures/confusion_matrix.png', dpi=300)
    plt.show()




    ########################
    # Optimizing the precision and recall of a classification model
    #######################
    from sklearn.metrics import precision_score, recall_score, f1_score

    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_test_pred))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_test_pred))
    print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_test_pred))





    #Evaluation 2: Crass val score on accuracy and roc_auc

    from sklearn.cross_validation import StratifiedKFold
    #we chose folds =3 is just for create the bad rsult of classifer,
    # because we want to see the different in ROC pic
    # usually n_foldes=10
    cv = StratifiedKFold(y_train, n_folds=10, random_state=1)


    #plot the roc curve
    from sklearn.metrics import roc_curve, auc
    from scipy import interp


    fig = plt.figure(figsize=(7, 5))

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i, (train, test) in enumerate(cv):
        probas = lr.fit(X_train[train],
                             y_train[train]).predict_proba(X_train[test])

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

    #Precision-Recall for muti- classifier with different threshold ?
    #http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
    from sklearn.multiclass import OneVsRestClassifier


    y_test_2 = np.array([y_test, (y_test-1)*-1], np.int32).T
    print((y_test_2.shape))

    y_train_2 = np.array([y_train, (y_train-1)*-1], np.int32).T
    print((y_train_2.shape))


    # setup plot details
    from itertools import cycle
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    lw = 2

    # Run classifier
    classifier = OneVsRestClassifier(LogisticRegression(random_state=0))
    y_score = classifier.fit(X_train, y_train_2).decision_function(X_test)

    print(y_score)
    n_classes = y_train_2.shape[1]
    print(n_classes)

    # Compute Precision-Recall and plot curve
    precision = dict()
    recall = dict()
    thresholds = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], thresholds[i] = precision_recall_curve(y_test_2[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(y_test_2[:, i], y_score[:, i])

    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], thresholds["micro"] = precision_recall_curve(y_test_2.ravel(),
        y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test_2, y_score,
                                                         average="micro")

    '''
    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall[0], precision[0], lw=lw, color='navy',
             label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
    plt.legend(loc="lower left")
    plt.show()
    '''
    # Plot Precision-Recall curve for each class
    plt.clf()
    plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
             label='micro-average Precision-recall curve (area = {0:0.2f})'
                   ''.format(average_precision["micro"]))
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=lw,
                 label='Precision-recall curve of class {0} (area = {1:0.2f})'
                       ''.format(i, average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(loc="lower right")
    plt.show()

    #################
    # cross_val: accuracy
    ###################

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(estimator=lr,
                                 X=X_train,
                                 y=y_train,
                                 cv=10,
                                 #scoring='roc_auc', #0.83 (+/- 0.01)
                                 scoring='accuracy',#0.77 (+/- 0.01)
                                 #scoring='roc_auc',
                                 n_jobs=-1)

    print("Accuracy: %0.2f (+/- %0.2f) [LogisticRegression]"
                   % (scores.mean(), scores.std()))



    #################
    # cross_val: roc_auc
    ###################

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(estimator=lr,
                                 X=X_train,
                                 y=y_train,
                                 cv=10,
                                 scoring='roc_auc', #0.83 (+/- 0.01)
                                 #scoring='accuracy',#0.77 (+/- 0.01)
                                 #scoring='roc_auc',
                                 n_jobs=-1)

    print("roc_auc: %0.2f (+/- %0.2f) [LogisticRegression]"
                   % (scores.mean(), scores.std()))


    #Evaluation 3: learning cruve for overfix and validation cruve for parameter.
    ###########################
    # get the learning cruve
    ##########################
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, test_scores =\
                learning_curve(estimator=lr,#estimator=pipe_lr,
                X=X_train,
                y=y_train,
                train_sizes=np.linspace(0.1, 1.0, 10),
                cv=10,
                n_jobs=-1)

    #plot it out
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    print("learning_curve train: %0.2f (+/- %0.2f) [AdaBoostClassifiers]"
          % ( np.asarray(train_mean).mean(),  np.asarray(train_std).mean() ))

    print("learning_curve test: %0.2f (+/- %0.2f) [AdaBoostClassifiers]"
          % ( np.asarray(test_mean).mean(),  np.asarray(test_std).mean() ))

    plt.plot(train_sizes, train_mean,
             color='blue', marker='o',
             markersize=5, label='training accuracy')

    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy')

    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.5, 1.0])
    plt.tight_layout()
    # plt.savefig('./figures/learning_curve.png', dpi=300)
    plt.show()



    #######################
    # Addressing overfit and underfitting with validation curves
    ########################
    from sklearn.model_selection import validation_curve

    # set the range of parameter we want to check: C in LogisticRegression
    param_range = [0.0001,0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    #param_range = [100,500,1000,5000]
    train_scores, test_scores = validation_curve(
                    #estimator=pipe_lr,
                    estimator=lr,
                    X=X_train,
                    y=y_train,
                    param_name='C',#C in LogisticRegression
                    #param_name='n_estimators',
                    param_range=param_range,
                    cv=10,
                    n_jobs=-1
                    )

    #plot it out
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(param_range, train_mean,
             color='blue', marker='o',
             markersize=5, label='training accuracy')

    plt.fill_between(param_range, train_mean + train_std,
                     train_mean - train_std, alpha=0.15,
                     color='blue')

    plt.plot(param_range, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy')

    plt.fill_between(param_range,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xscale('log')
    plt.legend(loc='lower right')
    plt.xlabel('C in LogisticRegression')
    plt.ylabel('Accuracy')
    plt.ylim([0.6, 1.0])
    #plt.tight_layout()
    # plt.savefig('./figures/validation_curve.png', dpi=300)
    plt.show()


    return 0



def adaptive_boosting(X,y,X_train,y_train,X_test,y_test):

    ######################
    #adaptive boosting
    #####################
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier



    tree = DecisionTreeClassifier(criterion='entropy',
                                  max_depth=1
                                  #min_samples_split=0.9 ,
                                  #max_features=50
                                 )

    clf = AdaBoostClassifier(base_estimator=tree,
                             n_estimators=500,
                             learning_rate=0.1,
                             random_state=0)


    #################
    # cross_val: roc_auc
    ###################

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(estimator=clf,
                                 X=X_train,
                                 y=y_train,
                                 cv=10,
                                 #scoring='roc_auc', #0.83 (+/- 0.01)
                                 #scoring='accuracy',#0.77 (+/- 0.01)
                                 scoring='roc_auc',
                                 n_jobs=-1)

    print("ROC AUC: %0.2f (+/- %0.2f) [AdaBoostClassifiers]"
                   % (scores.mean(), scores.std()))



    #################
    # cross_val: accuracy
    ###################

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(estimator=clf,
                                 X=X_train,
                                 y=y_train,
                                 cv=10,
                                 #scoring='roc_auc', #0.83 (+/- 0.01)
                                 scoring='accuracy',#0.77 (+/- 0.01)
                                 #scoring='roc_auc',
                                 n_jobs=-1)

    print("Accuracy: %0.2f (+/- %0.2f) [AdaBoostClassifiers]"
                   % (scores.mean(), scores.std()))


    ##################
    # poor score , but take a look
    #################
    from sklearn.metrics import accuracy_score

    clf = clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    clf_train = accuracy_score(y_train, y_train_pred)
    clf_test = accuracy_score(y_test, y_test_pred)
    print('AdaBoost train/test accuracies %.3f/%.3f'
          % (clf_train, clf_test))



    ###########################
    # get the learning cruve
    ##########################
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, test_scores =\
                learning_curve(estimator=clf,#estimator=pipe_lr,
                X=X_train,
                y=y_train,
                train_sizes=np.linspace(0.1, 1.0, 10),
                cv=10,
                n_jobs=-1)




    #plot it out
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    print("learning_curve train: %0.2f (+/- %0.2f) [AdaBoostClassifiers]"
          % ( np.asarray(train_mean).mean(),  np.asarray(train_std).mean() ))

    print("learning_curve test: %0.2f (+/- %0.2f) [AdaBoostClassifiers]"
          % ( np.asarray(test_mean).mean(),  np.asarray(test_std).mean() ))

    plt.plot(train_sizes, train_mean,
             color='blue', marker='o',
             markersize=5, label='training accuracy')

    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy')

    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.5, 1.0])
    #plt.tight_layout()
    # plt.savefig('./figures/learning_curve.png', dpi=300)
    plt.show()





    #######################
    # Addressing overfit and underfitting with validation curves
    ########################
    from sklearn.model_selection import validation_curve

    # set the range of parameter we want to check: C in LogisticRegression
    #param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    param_range = [100,500,1000,5000]
    train_scores, test_scores = validation_curve(
                    #estimator=pipe_lr,
                    estimator=clf,
                    X=X_train,
                    y=y_train,
                    #param_name='clf__C',#C in LogisticRegression
                    param_name='n_estimators',
                    param_range=param_range,
                    cv=10,
                    n_jobs=-1
                    )


    #plot it out
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(param_range, train_mean,
             color='blue', marker='o',
             markersize=5, label='training accuracy')

    plt.fill_between(param_range, train_mean + train_std,
                     train_mean - train_std, alpha=0.15,
                     color='blue')

    plt.plot(param_range, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy')

    plt.fill_between(param_range,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xscale('log')
    plt.legend(loc='lower right')
    plt.xlabel('n_estimators')
    plt.ylabel('Accuracy')
    plt.ylim([0.6, 1.0])
    #plt.tight_layout()
    # plt.savefig('./figures/validation_curve.png', dpi=300)
    plt.show()


    return 0

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

    #X=feature_preprocessing(X)

    #transform y label M,B to 1,0
    le = LabelEncoder()
    y = le.fit_transform(y)
    le.transform(['M', 'B'])
    #print(y)

    ############
    # separate train, test set
    #############
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)


    print("Data ready")



    #rbf_svc(X,y,X_train,y_train,X_test,y_test)
    #logisticRegression(X,y,X_train,y_train,X_test,y_test)
    adaptive_boosting(X,y,X_train,y_train,X_test,y_test)


    return 0


main()