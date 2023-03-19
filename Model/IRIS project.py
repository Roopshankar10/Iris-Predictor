#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
df=pd.read_csv("E:\Data Science\IRIS.csv")
df


# In[3]:


x=df.drop(['species'],axis=1)
x


# In[15]:



y=df['species']

y



# In[5]:


knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(x,y)
knn.score(x,y)


# In[6]:


logreg=LogisticRegression()
logreg.fit(x,y)
logreg.score(x,y)


# In[7]:


knn.predict([[6,3,5,6]])


# In[8]:


knn.predict([[6,3,4,2]])


# In[9]:


import pickle
with open('IRIS.pickle','wb') as f:
    pickle.dump(knn,f)


# In[10]:


import json
columns = {
    'data_columns' : [col.lower() for col in x.columns]
}
with open("iriscolumns.json","w") as f:
    f.write(json.dumps(columns))


# In[16]:


tar={
    "target_names":[c for c in y]
}
with open("species.josn",'w') as f:
    f.write(json.dumps(tar))


# In[20]:


def predict(sl,sw,pl,pw):
    arr=np.zeros(4)
    arr[0]=sl
    arr[1]=sw
    arr[2]=pl
    arr[3]=pw
    out=knn.predict([arr])
    st=""
    if out =='Iris-versicolor':
        st=y[0]
    elif out =='Iris-virginica':
        st=y[140]
    else:
        st=y[79]
    return st


# In[21]:


predict(6.2,3.4,5.4,2.3)


# In[ ]:




