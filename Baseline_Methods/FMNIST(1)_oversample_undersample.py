#!/usr/bin/env python
# coding: utf-8

# In[8]:


from utils import mnist_reader
import pandas as pd
X, y= mnist_reader.load_mnist('C:/Users/Devashi Jain/Desktop/IIIT-D/Stastical Machine Learning/Assignment1/data/fashion', kind='train')


# In[9]:


X=pd.DataFrame(X)


# In[10]:


X


# In[11]:


y=pd.DataFrame(y)


# In[17]:


y[0]


# In[24]:


data=pd.concat([X,y],axis=1,ignore_index=True)


# In[25]:


data


# In[26]:


modified_fmnist=[]
for i in range(len(X)):
    if y[0][i]==0 or y[0][i]==2:
        modified_fmnist.append(data.loc[i][:])
    elif y[0][i]== 3 or y[0][i]==1:
        modified_fmnist.append(data.loc[i][:])


# In[27]:


modified_fmnist=pd.DataFrame(modified_fmnist)


# In[28]:


modified_fmnist


# In[29]:


modified_fmnist.reset_index(inplace=True)
modified_fmnist.drop(['index'],axis=1)
fmnist=modified_fmnist


# In[31]:


for i in range(len(modified_fmnist)):
    if modified_fmnist[784][i]==1 or modified_fmnist[784][i]==3:
        modified_fmnist[784][i]=1
    elif modified_fmnist[784][i]==0 or modified_fmnist[784][i]==2:
        modified_fmnist[784][i]=2


# In[32]:


modified_fmnist.to_csv("fmnist.csv",index=False)


# # Oversampling using SMOTE

# In[34]:


from imblearn.over_sampling import SMOTE
smt = SMOTE()
X_train, y_train = smt.fit_sample(modified_fmnist.iloc[:,0:783],modified_fmnist[784])


# In[35]:


from sklearn.model_selection import train_test_split
X_t, X_test, y_t, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# # Random Forest Classifier
# 

# In[36]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
clf.fit(X_t, y_t) 
y_pred=clf.predict(X_test)


# In[37]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test, y_pred)


# In[38]:


acc


# # Gaussian Naive Bayes

# In[40]:


from sklearn.naive_bayes import GaussianNB
clf_nb = GaussianNB()
clf_nb.fit(X_t, y_t)
y_pred_nb=clf_nb.predict(X_test)
acc_nb=accuracy_score(y_test, y_pred_nb)
print(acc_nb)


# # svm

# In[42]:


from sklearn import svm
clf_svm = svm.SVC(gamma='scale')
clf_svm.fit(X_t, y_t)
y_pred_svm=clf_svm.predict(X_test)
acc_svm=accuracy_score(y_test, y_pred_svm)
print(acc_svm)


# In[43]:


# Logistic Regression


# In[44]:


from sklearn.linear_model import LogisticRegression
clf_lr = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_t, y_t)
y_pred_lr=clf_lr.predict(X_test)
acc_lr=accuracy_score(y_test, y_pred_lr)
print(acc_lr)


# In[45]:


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


# In[47]:


import numpy as np
Confusion_Matrix(y_test, y_pred)
Confusion_Matrix(y_test, y_pred_nb)
Confusion_Matrix(y_test, y_pred_svm)
Confusion_Matrix(y_test, y_pred_lr)


# In[48]:


# UnderSampling using NearMiss


# In[52]:


from imblearn.under_sampling import NearMiss
nr = NearMiss()
X_train, y_train = nr.fit_sample(modified_fmnist.iloc[:,0:783],modified_fmnist[784])
X_t_nr, X_test_nr, y_t_nr, y_test_nr = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# In[53]:


clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
clf.fit(X_t, y_t) 
y_pred_nr=clf.predict(X_test_nr)
acc_nr=accuracy_score(y_test_nr, y_pred_nr)
print(acc_nr)


# In[54]:


clf_nb = GaussianNB()
clf_nb.fit(X_t_nr, y_t_nr)
y_pred_nb_nr=clf_nb.predict(X_test_nr)
acc_nb_nr=accuracy_score(y_test_nr, y_pred_nb_nr)
print(acc_nb_nr)


# In[55]:


clf_svm = svm.SVC(gamma='scale')
clf_svm.fit(X_t_nr, y_t_nr)
y_pred_svm_nr=clf_svm.predict(X_test_nr)
acc_svm_nr=accuracy_score(y_test_nr, y_pred_svm_nr)
print(acc_svm_nr)


# In[56]:


clf_lr = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_t_nr, y_t_nr)
y_pred_lr_nr=clf_lr.predict(X_test_nr)
acc_lr_nr=accuracy_score(y_test_nr, y_pred_lr_nr)
print(acc_lr_nr)


# In[57]:


Confusion_Matrix(y_test_nr, y_pred_nr)
Confusion_Matrix(y_test_nr, y_pred_nb_nr)
Confusion_Matrix(y_test_nr, y_pred_svm_nr)
Confusion_Matrix(y_test_nr, y_pred_lr_nr)


# In[58]:


# Boosting


# In[69]:


#Boosting on oversampled data
from sklearn.ensemble import GradientBoostingClassifier
lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)
    gb_clf.fit(X_t, y_t)

    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_t, y_t)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))


# In[60]:


gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.75, max_features=2, max_depth=2, random_state=0)
gb_clf.fit(X_t, y_t)

print("Learning rate: ", learning_rate)
print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_t, y_t)))
print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))
y_pred_gradient_boosting_ov=gb_clf.predict(X_test)
Confusion_Matrix(y_test, y_pred_gradient_boosting_ov)


# In[61]:


#Boosting using Undersampled data
lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)
    gb_clf.fit(X_t_nr, y_t_nr)

    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_t_nr, y_t_nr)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test_nr, y_test_nr)))


# In[62]:


gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.75, max_features=2, max_depth=2, random_state=0)
gb_clf.fit(X_t_nr, y_t_nr)

print("Learning rate: ", learning_rate)
print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_t_nr, y_t_nr)))
print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test_nr, y_test_nr)))
y_pred_gradient_boosting_nr=gb_clf.predict(X_test_nr)
Confusion_Matrix(y_test_nr, y_pred_gradient_boosting_nr)


# In[63]:


# Bagging


# In[64]:


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


# In[65]:


#Bagging using oversampled data
Predicted_ov=[]
Predicted_ov.append(y_pred)
Predicted_ov.append(y_pred_nb)
Predicted_ov.append(y_pred_svm)
Predicted_ov.append(y_pred_lr)


# In[66]:


y_pred_ov_bagging=Hard_Predicted(Predicted_ov)
acc_bagging_ov=accuracy_score(y_test,y_pred_ov_bagging)
print(acc_bagging_ov)
Confusion_Matrix(y_test, y_pred_ov_bagging)


# In[67]:


#Bagging using Undersampled data
Predicted_ov_nr=[]
Predicted_ov_nr.append(y_pred_nr)
Predicted_ov_nr.append(y_pred_nb_nr)
Predicted_ov_nr.append(y_pred_svm_nr)
Predicted_ov_nr.append(y_pred_lr_nr)


# In[68]:


y_pred_un_bagging=Hard_Predicted(Predicted_ov_nr)
acc_bagging_un_nr=accuracy_score(y_test_nr,y_pred_un_bagging)
print(acc_bagging_un_nr)
Confusion_Matrix(y_test_nr, y_pred_un_bagging)


# In[ ]:




