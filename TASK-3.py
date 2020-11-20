#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Name- Kirtee Bakshi
TASK 3- CONDUCTING AN EXPLORATORY DATA ANALYSIS ON A SAMPLE SUPERSTORE


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


mydata=pd.read_csv('SampleSuperstore.csv')


# In[3]:


mydata.head()


# In[4]:


mydata.describe()


# In[5]:


mydata.info()


# In[6]:


mydata.isnull().sum()


# In[7]:


mydata.corr()


# In[8]:


mydata['Category'].unique()


# In[9]:


mydata['Category'].value_counts()


# In[10]:


mydata['State'].unique()


# In[11]:


fig, ax=plt.subplots(figsize=(25,10))
x=mydata['State']
y=mydata['Sales']
ax.set_xlabel('State')
ax.set_ylabel('Sales')
ax.set_title('State wise sales distribution')
fig.autofmt_xdate()
ax.bar(x,y)
plt.show()


# In[12]:


fig, ax=plt.subplots(figsize=(25,10))
x=mydata['State']
y=mydata['Profit']
ax.set_xlabel('State')
ax.set_ylabel('Profit')
ax.set_title('State wise profit distribution')
fig.autofmt_xdate()
ax.bar(x,y)
plt.show() 


# In[13]:


fig, ax=plt.subplots(figsize=(10,10))
x=mydata['Region']
y=mydata['Sales']
ax.set_xlabel('Region')
ax.set_ylabel('Sales')
ax.set_title('Region wise sales distribution')
fig.autofmt_xdate()
ax.bar(x,y)
plt.show() 


# In[14]:


fig, ax=plt.subplots(figsize=(10,5))
x=mydata['Region']
y=mydata['Profit']
ax.set_xlabel('Region')
ax.set_ylabel('Profit')
ax.set_title('Region wise Profit distribution')
fig.autofmt_xdate()
ax.bar(x,y)
plt.show() 


# In[15]:


fig, ax=plt.subplots(figsize=(10,5))
x=mydata['Category']
y=mydata['Sales']
ax.set_xlabel('Category')
ax.set_ylabel('Sales')
ax.set_title('Category wise Sales distribution')
fig.autofmt_xdate()
ax.bar(x,y)
plt.show() 


# In[16]:


fig, ax=plt.subplots(figsize=(10,5))
x=mydata['Category']
y=mydata['Profit']
ax.set_xlabel('Category')
ax.set_ylabel('Profit')
ax.set_title('Category wise Profit distribution')
fig.autofmt_xdate()
ax.bar(x,y)
plt.show() 


# In[ ]:


#we can see from the graphs that Florida has maximum number of sales but Florida profit is in negative.
#This means even if the state(Florida)is creating a large number of sales,its not generating profit.
#The east region is the weakest in both generating profit and sales.
#We see that furniture generates the lowest profit as its sales are very low.

