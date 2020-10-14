import numpy as np
import random
import os
import math 
import time

def find_files(prothem,spam):
    l_data=[]
    for file in os.listdir("../data/"+prothem+'/'):
        if file.endswith(".txt"):
            l_data.append([prothem,file,spam])
    return l_data

def prepare(clasfiles):
    l_data=[]
    for clasfile in clasfiles:
        topic_list=[]
        f=open("../data/"+clasfile[0]+"/"+clasfile[1],"r",errors='ignore')
        topic_list=[]
        for line in f:
            kappa = line.split()
            topic_list=topic_list +kappa
        #remove duplicates 
        topic_list = list(dict.fromkeys(topic_list))   
        f.close()
        l_data.append([topic_list,clasfile[2]])
    return l_data

def split(dataset,ratio):
    l_tr=[]
    l_de=[]
    l_te=[]
    for tokens in dataset:
        rand =random.random()
        if rand >= 1-ratio:
            l_tr.append(tokens)
        elif rand >= (1-ratio)/2:
            l_de.append(tokens)
        else:
            l_te.append(tokens)
    return l_tr,l_de,l_te

def choose_data(l_dataset,ratio):
    l_tr=[]
    for tokens in l_dataset:
        rand =random.random()
        if rand >= 1-ratio:
            l_tr.append(tokens)
    return l_tr

def precision_recall_f1(correct_list,prediction_list):
    """Returns precision,recall and f1 scores."""
    f_n=0
    f_p=0
    t_p=0
    for i in range(0,len(correct_list)):
        if correct_list[i]==prediction_list[i] and correct_list[i]==1:
            t_p=t_p+1
        elif correct_list[i]!=prediction_list[i] and correct_list[i]==1:
            f_n=f_n+1
        elif correct_list[i]!=prediction_list[i] and correct_list[i]==0:
            f_p=f_p+1
    precision=(t_p+1)/(t_p+f_p+4)
    recall=(t_p+1)/(t_p+f_n+4)
    f1=2*precision*recall/(precision+recall)
    return [precision,recall,f1]

def make_voc(dataset):
    """Returns all found tokens, how many times found, how many times found and being spam and spams generally sorted by Information Gain"""
    voc=[]
    voc_n=[]
    dataset= np.array(dataset)
    l_sp=0
    for text in dataset:
        for token in text[0]:
            try:
                where=voc.index(token)
            except:
                voc.append(token)
                voc_n.append([text[1],1])
            else:
                voc_n[where][0]=voc_n[where][0]+text[1]
                voc_n[where][1]=voc_n[where][1]+1
        l_sp = l_sp + text[1]
    voc,voc_n = sort_voc_by_IG(len(dataset),voc,voc_n,l_sp)
    return voc,voc_n,l_sp    


def make_vectors(dataset,voc):
    """Creates a (dataset x voc) vector declaring in [i][j] if token in j position of voc is found in position i of dataset."""
    table_list=np.zeros((len(dataset),len(voc)),dtype=bool)
    for i in range(len(dataset)):
        for token in dataset[i][0]:
            try:
                where = voc.index(token)
            except:
                continue
            else:
                table_list[i][where] = 1
    return table_list

def make_zero_one(dataset):
    zero_one=np.zeros((len(dataset)),dtype=bool)
    for i in range(len(dataset)):
        zero_one[i]=dataset[i][1]
    return zero_one

def H(prob):
    h=0
    for i in range(len(prob)):
        h= h+prob[i]*math.log(prob[i])
    h=-h
    return h

def IG(prob_c,prob_x,prob_c_x):
    ig=H(prob_c)
    for i in range(len(prob_x)):
        ig= ig-prob_x[i]*H(prob_c_x[i])
    return ig

def sort_voc_by_IG(len_dataset,voc,voc_n,l_sp):
    l_ham=len_dataset-l_sp
    p_spam= (l_sp+1)/(len_dataset+2)
    p_ham= (l_ham+1)/(len_dataset+2)
    n=[[0,i] for i in range(len(voc))]
    for i in range(len(voc)):
        p_x_0= (len_dataset-voc_n[i][1]+1)/(len_dataset+2)
        p_x_1= (voc_n[i][1]+1)/(len_dataset+2)
        p_x_1_c_0=(voc_n[i][1]-voc_n[i][0]+1)/(len_dataset+2)
        p_x_1_c_1=(voc_n[i][0]+1)/(len_dataset+2)
        p_x_0_c_0=(l_ham-voc_n[i][1]+voc_n[i][0]+1)/(len_dataset-voc_n[i][1]+2)
        p_x_0_c_1=(l_sp-voc_n[i][0]+1)/(len_dataset-voc_n[i][1]+2)
        n[i][0]= IG([p_ham,p_spam],[p_x_0,p_x_1],[[p_x_0_c_0,p_x_0_c_1],[p_x_1_c_0,p_x_1_c_1]])
    n=sorted(n,key=lambda l:l[0], reverse = True) 
    voc_list1=[]
    voc_list2=[]
    for j in range (len(n)):
        index=n[j][1]
        voc_list1.append(voc[index])
        voc_list2.append([voc_n[index][0],voc_n[index][1]])
    return voc_list1,voc_list2 


def accuracy_score(correct_list,predict_list):
    correct=0
    for i in range(0,len(correct_list)):
        if correct_list[i][1]==predict_list[i][1]:
            correct=correct+1
    return correct/len(correct_list)

def from_vectors_to_ints(X,Y):
    voc_n=np.zeros((len(X[0]),2),dtype=int)
    for i in range(len(X)):
        for j in range(len(X[0])):
            if X[i][j]==1:
                voc_n[j][0]+=Y[i]
                voc_n[j][1]+=1
    l_sp = sum(Y)
    return voc_n,l_sp