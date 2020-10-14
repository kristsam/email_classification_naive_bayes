#Bernoulli Naive Bayes implementation using matplotlib for plots.

#pip install matplotlib

import math 
import random
import matplotlib.pyplot as plt
from interface import *

class Naive_Bayes_Manual:

    def __init__(self,dataset,training,development,test,vocabulary,voc_len,vocabulary_N,spams):
        self.dataset=dataset
        self.training = training
        self.development = development
        self.test = test
        self.vocabulary = vocabulary
        self.vocabulary_N = vocabulary_N
        self.spams = spams
        self.vocabulary,self.vocabularyN,Naive_Bayes_Manual.prob_spam,Naive_Bayes_Manual.prob_ham= self.choose_training_by_voc_len(voc_len,vocabulary,vocabulary_N,spams)
        
    def fit(self,voc,voc_n,times,spams):
        prob_spam=[]
        prob_ham=[]
        hams=times-spams
        for i in range (len(voc)):
            sp_x_1= math.log((voc_n[i][0]+1)/(spams+2))
            sp_x_0= math.log((spams-voc_n[i][0]+1)/(spams+2))
            ha_x_1= math.log((voc_n[i][1]-voc_n[i][0]+1)/(hams+2))
            ha_x_0= math.log((hams-voc_n[i][1]+voc_n[i][0]+1)/(hams+2))
            prob_spam.append([sp_x_1,sp_x_0])
            prob_ham.append([ha_x_1,ha_x_0])
        return prob_spam,prob_ham

    def predict(self,l_dataset,voc,voc_n,total,spams,prob_spam,prob_ham):
        predicted=[]
        hams= total- spams
        for tokens in l_dataset:
            spam=0
            ham=0
            for i in range(len(voc)):
                if voc[i] in tokens[0]:
                    spam += prob_spam[i][0]
                    ham += prob_ham[i][0]
                else:
                    spam += prob_spam[i][1]
                    ham += prob_ham[i][1]
            spam += math.log((spams+1)/(total+2))
            ham += math.log((hams+1)/(total+2))
            if spam>ham:
                predicted.append([tokens[0],1])
            else:
                predicted.append([tokens[0],0])
        return predicted

    def choose_training_by_voc_len(self,p_voc_len, p_voc,p_voc_n, spams):
        """Decides which should be the length of the vocabulary by calculating scores in development set."""
        voc_b_c=[]
        voc_b_c_N=[]
        for i in range(len(p_voc_len)):
            vo=p_voc[:p_voc_len[i]]
            voN=p_voc_n[:p_voc_len[i]]
            voc_b_c.append(vo)
            voc_b_c_N.append(voN)
        max_score=-1
        for i in range (len(p_voc_len)):
            prob_spam,prob_ham = self.fit(voc_b_c[i],voc_b_c_N[i],len(self.training),spams)
            development_predicted = self.predict(self.development,voc_b_c[i],voc_b_c_N[i],len(self.training),spams,prob_spam,prob_ham)
            score=accuracy_score(self.development,development_predicted)
            if score>max_score:
                max_p_s=prob_spam
                max_p_h=prob_ham
                vocabulary=voc_b_c[i]
                vocabularyN=voc_b_c_N[i]
                max_score=score
        return vocabulary,vocabularyN,max_p_s,max_p_h

    def multiple_examples(self,runs):
        """Produces error,precision,recall and f1 rate diagrams."""
        tr_pr=[]
        tr_re=[]
        tr_f1=[]
        te_pr=[]
        te_re=[]
        te_f1=[]
        training_errors=[]
        test_errors=[]
        x1=[]
        x2=[]
        X_train=make_vectors(self.training,self.vocabulary)
        Y_train=make_zero_one(self.training)
        voc=self.vocabulary
        for i in range (runs):
            chosen_training = self.training[:int(len(self.training)*(i+1)/runs)]
            Y_t = Y_train[:int(len(self.training)*(i+1)/runs)]
            X_t = X_train[:int(len(self.training)*(i+1)/runs)]
            voc_n,spams= from_vectors_to_ints(X_t,Y_t)
            prob_spam,prob_ham = self.fit(voc,voc_n,len(chosen_training),spams)
            training_predicted = self.predict(chosen_training,voc,voc_n,len(chosen_training),spams,prob_spam,prob_ham)
            training_errors.append((1-accuracy_score(chosen_training,training_predicted))*100)
            training_p_r_f = precision_recall_f1(make_zero_one(chosen_training),(make_zero_one(training_predicted)))
            test_predicted = self.predict(self.test,voc,voc_n,len(chosen_training),spams,prob_spam,prob_ham)
            test_errors.append((1-accuracy_score(self.test,test_predicted))*100)
            test_p_r_f = precision_recall_f1(make_zero_one(self.test),make_zero_one(test_predicted))
            x1.append((i+1)/runs*100)
            x2.append((i+1)/runs*100)
            tr_pr.append(training_p_r_f[0])
            tr_re.append(training_p_r_f[1])
            tr_f1.append(training_p_r_f[2])
            te_pr.append(test_p_r_f[0])
            te_re.append(test_p_r_f[1])
            te_f1.append(test_p_r_f[2])

        fig = plt.figure()
        
        fig.suptitle("Naive Bayes manual, examples="+str(runs)+", tokens="+str(len(self.vocabulary)))

        plt.subplot(4, 1, 1)
        plt.plot(x1,training_errors, label="training data")
        plt.plot(x2,test_errors, label="testing data")
        plt.ylabel('error %')
        plt.xlabel('training examples %')
        plt.axis([0, 100, 0, 100])

        plt.subplot(4, 1, 2)
        plt.plot(x1,tr_pr, label="training data")
        plt.plot(x2,te_pr, label="testing data")
        plt.ylabel('precision')
        plt.xlabel('training examples %')
        
        plt.subplot(4, 1, 3)
        plt.plot(x1,tr_re, label="training data")
        plt.plot(x2,te_re, label="testing data")
        plt.ylabel('recall')
        plt.xlabel('training examples %')

        plt.subplot(4, 1, 4)
        plt.plot(x1,tr_f1, label="training data")
        plt.plot(x2,te_f1, label="testing data")
        plt.ylabel('f1')
        plt.xlabel('training examples %')

        plt.legend()
        fig.savefig("../diagrams/Naive_Bayes_Manual.png")
        return

    def test_score(self):
        """Calculates and prints score% of test dataset calculated by fited training dataset.
            
            Returns

            score:float
                Testing dataset prediction score."""
        test_predicted = self.predict(self.test,self.vocabulary,self.vocabularyN,len(self.training),self.spams,self.prob_spam,self.prob_ham)
        score=accuracy_score(self.test,test_predicted)
        print("Prediction score in test set -using manual Naive_Bayes- is "+str(score*100)+" %.")
        return score
