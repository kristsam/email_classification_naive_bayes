#Bernoulli Naive Bayes implementation based on sklearn library for ML and matplotlib for plots.

#pip install -U scikit-learn

from sklearn.naive_bayes import BernoulliNB
from interface import *
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

class Naive_Bayes_Sklearn:
    
    clf = BernoulliNB()

    def __init__(self,dataset,training,development,test,vocabulary,voc_len):
        self.dataset=dataset
        self.training = training
        self.development = development
        self.test = test
        self.vocabulary=vocabulary
        self.voc_len = voc_len
        Naive_Bayes_Sklearn.Y_train = make_zero_one(self.training)
        self.vocabulary,Naive_Bayes_Sklearn.X_train = self.choose_training_by_voc_len(voc_len,vocabulary)
        Naive_Bayes_Sklearn.X_test = make_vectors(self.test,self.vocabulary)
        Naive_Bayes_Sklearn.Y_test = make_zero_one(self.test)    


    def choose_training_by_voc_len(self, p_voc_len, p_voc ):
        """Decides which should be the length of the vocabulary by calculating scores in development set."""
        voc_b_c=[]
        for i in range(len(p_voc_len)):
            vo=p_voc[:p_voc_len[i]]
            voc_b_c.append(vo)
        max_score=-1
        Y_dev = make_zero_one(self.development)
        for i in range (len(p_voc_len)):
            X = make_vectors(self.training,voc_b_c[i])
            X_dev = make_vectors(self.development,voc_b_c[i])
            self.clf.fit(X,self.Y_train)
            development_predicted = self.clf.predict(X_dev)
            score=self.clf.score(X_dev,Y_dev)
            if score>max_score:
                vocabulary=voc_b_c[i]
                max_score=score
        X = make_vectors(self.training,vocabulary)
        return vocabulary , X

    def multiple_examples(self,n):
        """Produces error,precision,recall and f1 rate diagrams."""
        X = make_vectors(self.dataset,self.vocabulary)
        Y= make_zero_one (self.dataset)
        train_sizes, train_scores, test_scores = learning_curve(self.clf, X, Y, train_sizes=np.linspace(1/n, 1, n), cv=None)
        pre_tr_list=[]
        re_tr_list=[]
        f1_tr_list=[]
        pre_te_list=[] 
        re_te_list=[]
        f1_te_list=[]
        voc=self.vocabulary    
        for i in range(n):
            chosen_training = self.training[:int(len(self.training)*(i+1)/n)]
            X_tr = make_vectors(chosen_training,voc)
            Y_tr_correct = make_zero_one(chosen_training)
            self.clf.fit(X_tr,Y_tr_correct)
            Y_tr_pred = self.clf.predict(X_tr)
            tr=precision_recall_f1(Y_tr_correct,Y_tr_pred)
            pre_tr_list.append(tr[0])
            re_tr_list.append(tr[1])
            f1_tr_list.append(tr[2])
            X_te = make_vectors(self.test,voc)
            Y_te_correct = make_zero_one(self.test)
            Y_te_pred = self.clf.predict(X_te)
            te=precision_recall_f1(Y_te_correct,Y_te_pred)
            pre_te_list.append(te[0])
            re_te_list.append(te[1])
            f1_te_list.append(te[2])

        train_sizes=[i/train_sizes[-1]*100 for i in train_sizes]
        train_scores=[(1-np.mean(i))*100 for i in train_scores]
        test_scores=[(1-np.mean(i))*100 for i in test_scores]
    
        fig = plt.figure()

        fig.suptitle("Naive Bayes sklearn, examples="+str(n)+", tokens="+str(len(self.vocabulary)))
        plt.grid()
    
        plt.subplot(4,1,1)
        plt.plot(train_sizes, train_scores, label="training data")
        plt.plot(train_sizes,test_scores, label="testing data")
        plt.ylabel('error %')
        plt.xlabel('training examples %')
        plt.axis([0, 100, 0, 100])

        plt.subplot(4, 1, 2)
        plt.plot(train_sizes,pre_tr_list, label="training data")
        plt.plot(train_sizes,pre_te_list, label="testing data")
        plt.ylabel('precision')
        plt.xlabel('training examples %')
        
        plt.subplot(4, 1, 3)
        plt.plot(train_sizes,re_tr_list, label="training data")
        plt.plot(train_sizes,re_te_list, label="testing data")
        plt.ylabel('recall')
        plt.xlabel('training examples %')

        plt.subplot(4, 1, 4)
        plt.plot(train_sizes,f1_tr_list, label="training data")
        plt.plot(train_sizes,f1_te_list, label="testing data")
        plt.ylabel('f1')
        plt.xlabel('training examples %')

        plt.legend()
        fig.savefig("../diagrams/Naive_Bayes_Sklearn.png")
        return

    def test_score(self):
        """Calculates and prints score% of test dataset calculated by fited training dataset.
        
            Returns

            score:float
                Testing dataset prediction score."""
        self.clf.fit(self.X_train, self.Y_train)
        predicted_test=self.clf.predict(self.X_test)
        score=self.clf.score(self.X_test,self.Y_test)
        print("Prediction score in test set -using Naive_Bayes in Sklearn library- is "+str(score*100)+" %.")
        return score
