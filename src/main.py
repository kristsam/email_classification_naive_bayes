from interface import *
from naive_bayes_manual import Naive_Bayes_Manual
from naive_bayes_sklearn import Naive_Bayes_Sklearn

start = time.time()

directories=["enron1/spam1","enron1/ham1","enron1/spam2","enron1/ham2","enron1/spam3","enron1/ham3","enron2/spam","enron2/ham"]
voc_len=[400,600,900,1200]

pure_data=[]
for i in range(len(directories)):
    pure_data += find_files(directories[i],(i+1)%2)
dataset = prepare(pure_data)
training,development,test = split(dataset,0.8)
vocabulary,voc_n,spams = make_voc(training)

n_b_s = Naive_Bayes_Sklearn(dataset,training,development,test,vocabulary,voc_len)
n_b_s.test_score()
n_b_s.multiple_examples(10)

n_b_m = Naive_Bayes_Manual(dataset,training,development,test,vocabulary,voc_len,voc_n,spams)
n_b_m.test_score()
n_b_m.multiple_examples(10)

print(time.time()-start)