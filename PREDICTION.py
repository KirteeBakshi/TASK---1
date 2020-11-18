#!/usr/bin/env python
# coding: utf-8

# THE SPARKS FOUNDATION
# NAME - KIRTEE BAKSHI 
# TASK 1- Prediction using supervised ML (Predict the percentage of a student based on the no of study hours)
# 

# In[17]:


#Importing all the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[18]:


# Reading data from remote link
URL="http://bit.ly/w-data"
data=pd.read_csv(URL)
print("Data imported successfully")


# In[19]:


data.head(10)


# In[20]:


data.tail(5)


# In[21]:


data.info()


# In[22]:


data.describe()


# In[23]:


# Plotting the distribution of scores
data.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Scores')
plt.xlabel('Hours studied')
plt.ylabel('Scores Obtained')
plt.show()


# In[24]:


# Dividing the data into "attributes" (inputs) and "labels" (outputs).
X=data.iloc[:,:1].values
y=data.iloc[:,1].values


# In[25]:


#split this data into training and test sets.
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0) 


# In[26]:


#trainning our algorithm
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
print ('Training complete')


# In[27]:


line=regressor.coef_* X + regressor.intercept_
plt.scatter(X,y)
plt.plot(X,line)
plt.show()


# In[28]:


print(X_test)
y_pred=regressor.predict(X_test)
print(y_pred)


# In[29]:


print(regressor.intercept_)


# In[30]:


df=pd.DataFrame({'Actual':y_test, 'Predicted': y_pred})
df


# In[31]:


plt.scatter(X,y,color='red')
plt.plot(X_test,y_pred,color='blue')


# In[32]:


print("Score for studying for 9.25 hours is",regressor.predict([[9.25]]))


# In[33]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:




