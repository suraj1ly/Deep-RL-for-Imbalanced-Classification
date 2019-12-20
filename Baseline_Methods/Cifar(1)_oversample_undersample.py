#!/usr/bin/env python
# coding: utf-8

# # Loading Cifar 10

# In[1]:


import os
import numpy as np
import pandas as pd
from six.moves import cPickle as pickle


# In[2]:


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        x = pickle.load(fo, encoding='bytes')
    return x[b'data'],x[b'labels']


# In[3]:


path="C:/Users/Devashi Jain/Desktop/IIIT-D/Stastical Machine Learning/Assignment3/train"
TrainingSet= pd.DataFrame()
Labels=pd.DataFrame()
for folder in os.listdir(path):
    file=os.path.join(path,folder)
    Data,Label=unpickle(file)
    Data=pd.DataFrame(Data)
    Labels=Labels.append(Label)
    TrainingSet=TrainingSet.append(Data)
TrainingSet.index=range(len(TrainingSet))   


# In[4]:


import numpy as np
def RGBtoGray(dataframe):
    dataframe=dataframe.values
    grayscale=np.zeros((len(dataframe),1024))
    for i in range(len(dataframe)):
        for j in range(1024):
            grayscale[i][j]=0.299*dataframe[i][j]+0.587*dataframe[i][j+1024]+0.114*dataframe[i][j+2048]
    return grayscale  


# In[5]:


gray=RGBtoGray(TrainingSet)


# In[6]:


Train=pd.DataFrame(gray)
Labels.index=range(len(Labels))
Labels.columns=['Label']
Train=pd.concat([Train,Labels],axis=1)


# In[7]:


path="C:/Users/Devashi Jain/Desktop/IIIT-D/Stastical Machine Learning/Assignment3/test"
TestingSet= pd.DataFrame()
TestLabels=pd.DataFrame()
for folder in os.listdir(path):
    file=os.path.join(path,folder)
    print(file)
    TestData,Labels=unpickle(file)
    TestData=pd.DataFrame(TestData)
    TestLabels=TestLabels.append(Labels)
    TestingSet=TestingSet.append(TestData)
TestingSet.index=range(len(TestingSet))   


# In[8]:


grayTest=RGBtoGray(TestingSet)


# In[9]:


Test=pd.DataFrame(grayTest)
TestLabels.index=range(len(TestLabels))
TestLabels.columns=['Label']
Test=pd.concat([Test,TestLabels],axis=1)


# In[68]:


modified_cifar=[]
for i in range(len(Train)):
    if Train['Label'][i]==1:
        modified_cifar.append(Train.loc[i][:])
    elif Train['Label'][i]== 3 or Train['Label'][i]==4 or Train['Label'][i]==5 or Train['Label'][i]==6:
        modified_cifar.append(Train.loc[i][:])


# In[72]:


modified_cifar=pd.DataFrame(modified_cifar)


# In[73]:


modified_cifar


# In[76]:


modified_cifar.reset_index(inplace=True)


# In[77]:


modified_cifar.drop(['index'],axis=1)


# In[78]:


modified_cifar['Label'].unique()


# In[79]:


cifar=modified_cifar


# In[55]:


cifar


# # Imbalancing the Dataset

# In[80]:


for i in range(len(modified_cifar)):
    if modified_cifar['Label'][i]==3 or modified_cifar['Label'][i]==4 or modified_cifar['Label'][i]==5 or modified_cifar['Label'][i]==6:
        modified_cifar['Label'][i]=2


# In[91]:


modified_cifar.to_csv("cifar.csv",index=False)


# # Oversampling using SMOTE

# In[81]:


from imblearn.over_sampling import SMOTE
smt = SMOTE()
X_train, y_train = smt.fit_sample(modified_cifar.iloc[:,0:1023],modified_cifar['Label'])


# In[82]:


from sklearn.model_selection import train_test_split
X_t, X_test, y_t, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# # Random Forest Classifier

# In[83]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
clf.fit(X_t, y_t) 
y_pred=clf.predict(X_test)


# In[84]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test, y_pred)


# In[85]:


acc


# # Gaussian Naive Bayes

# In[88]:


from sklearn.naive_bayes import GaussianNB
clf_nb = GaussianNB()
clf_nb.fit(X_t, y_t)
y_pred_nb=clf_nb.predict(X_test)
acc_nb=accuracy_score(y_test, y_pred_nb)
print(acc_nb)


# # SVM

# In[89]:


from sklearn import svm
clf_svm = svm.SVC(gamma='scale')
clf_svm.fit(X_t, y_t)
y_pred_svm=clf_svm.predict(X_test)


# In[90]:


acc_svm=accuracy_score(y_test, y_pred_svm)
print(acc_svm)


# # Logistic Regression

# In[92]:


from sklearn.linear_model import LogisticRegression
clf_lr = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_t, y_t)
y_pred_lr=clf_lr.predict(X_test)
acc_lr=accuracy_score(y_test, y_pred_lr)
print(acc_lr)


# In[101]:


from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
def Confusion_Matrix(y_true,y_pred):
    data = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(data, columns=np.unique(y_true), index = np.unique(y_true))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (10,7))
    sn.set(font_scale=1.4)#for label size
    sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size


# In[103]:


Confusion_Matrix(y_test, y_pred)
Confusion_Matrix(y_test, y_pred_nb)
Confusion_Matrix(y_test, y_pred_svm)
Confusion_Matrix(y_test, y_pred_lr)


# # UnderSampling using NearMiss

# In[93]:


from imblearn.under_sampling import NearMiss
nr = NearMiss()
X_train, y_train = nr.fit_sample(modified_cifar.iloc[:,0:1023],modified_cifar['Label'])
X_t_nr, X_test_nr, y_t_nr, y_test_nr = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# In[95]:


clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
clf.fit(X_t, y_t) 
y_pred_nr=clf.predict(X_test_nr)
acc_nr=accuracy_score(y_test_nr, y_pred_nr)
print(acc_nr)


# In[96]:


clf_nb = GaussianNB()
clf_nb.fit(X_t_nr, y_t_nr)
y_pred_nb_nr=clf_nb.predict(X_test_nr)
acc_nb_nr=accuracy_score(y_test_nr, y_pred_nb_nr)
print(acc_nb_nr)


# In[97]:


clf_svm = svm.SVC(gamma='scale')
clf_svm.fit(X_t_nr, y_t_nr)
y_pred_svm_nr=clf_svm.predict(X_test_nr)
acc_svm_nr=accuracy_score(y_test_nr, y_pred_svm_nr)
print(acc_svm_nr)


# In[98]:


clf_lr = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_t_nr, y_t_nr)
y_pred_lr_nr=clf_lr.predict(X_test_nr)
acc_lr_nr=accuracy_score(y_test_nr, y_pred_lr_nr)
print(acc_lr_nr)


# In[104]:


Confusion_Matrix(y_test_nr, y_pred_nr)
Confusion_Matrix(y_test_nr, y_pred_nb_nr)
Confusion_Matrix(y_test_nr, y_pred_svm_nr)
Confusion_Matrix(y_test_nr, y_pred_lr_nr)


# # Boosting

# In[105]:


#Boosting on oversampled data
from sklearn.ensemble import GradientBoostingClassifier
lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)
    gb_clf.fit(X_t, y_t)

    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_t, y_t)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))


# In[107]:


gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.75, max_features=2, max_depth=2, random_state=0)
gb_clf.fit(X_t, y_t)

print("Learning rate: ", learning_rate)
print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_t, y_t)))
print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))
y_pred_gradient_boosting_ov=gb_clf.predict(X_test)
Confusion_Matrix(y_test, y_pred_gradient_boosting_ov)


# In[108]:


#Boosting using Undersampled data
lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)
    gb_clf.fit(X_t_nr, y_t_nr)

    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_t_nr, y_t_nr)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test_nr, y_test_nr)))


# In[109]:


gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.75, max_features=2, max_depth=2, random_state=0)
gb_clf.fit(X_t_nr, y_t_nr)

print("Learning rate: ", learning_rate)
print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_t_nr, y_t_nr)))
print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test_nr, y_test_nr)))
y_pred_gradient_boosting_nr=gb_clf.predict(X_test_nr)
Confusion_Matrix(y_test_nr, y_pred_gradient_boosting_nr)


# # Bagging

# In[113]:


import collections
import operator
def Hard_Predicted(Label):
    predicted_label=[]
    Label=pd.DataFrame(Label)
    row,col=Label.shape
    for i in range(col):
        counter=collections.Counter(Label[i])
        sorted_counter = sorted(counter.items(), key=operator.itemgetter(0))
        max_counter=max(counter.items(), key=operator.itemgetter(1))
        y={}
        for term in counter:
            if(counter[term]>=max_counter[1]):
                y[term]=counter[term]
        sorted_counter = sorted(y.items(), key=operator.itemgetter(0))
        predicted_label.append(sorted_counter[0][0])
    return (predicted_label)


# In[110]:


#Bagging using oversampled data
Predicted_ov=[]
Predicted_ov.append(y_pred)
Predicted_ov.append(y_pred_nb)
Predicted_ov.append(y_pred_svm)
Predicted_ov.append(y_pred_lr)


# In[116]:


y_pred_ov_bagging=Hard_Predicted(Predicted_ov)
acc_bagging_ov=accuracy_score(y_test,y_pred_ov_bagging)
print(acc_bagging_ov)
Confusion_Matrix(y_test, y_pred_ov_bagging)


# In[119]:


#Bagging using Undersampled data
Predicted_ov_nr=[]
Predicted_ov_nr.append(y_pred_nr)
Predicted_ov_nr.append(y_pred_nb_nr)
Predicted_ov_nr.append(y_pred_svm_nr)
Predicted_ov_nr.append(y_pred_lr_nr)


# In[120]:


y_pred_un_bagging=Hard_Predicted(Predicted_ov_nr)
acc_bagging_un_nr=accuracy_score(y_test_nr,y_pred_un_bagging)
print(acc_bagging_un_nr)
Confusion_Matrix(y_test_nr, y_pred_un_bagging)


# In[ ]:




