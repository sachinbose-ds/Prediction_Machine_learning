
# coding: utf-8

# # # Predicting Diabetes
# 

# # Import Libraries

# In[2]:


import pandas as pd    #pandas is dataframe library
import matplotlib.pyplot as plt  #plots data
import numpy as np     #numpy provides N dim object support

get_ipython().magic(u'matplotlib inline')


# # Load and review data

# In[3]:


df = pd.read_csv("pima-data.csv")


# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


df.tail()


# Check for null values

# In[7]:


df.isnull().values.any()


# Ploting the graph

# In[8]:


def plot_corr(df,size =11):
    """
    Function plots a graphical corrrelation matrix for each pair of column in the dataframe.
    
    Input:
        df:pandas DataFrame
        size:verticle and horizontal size of the plot
        
    Displays:
        matrix of correlation between column. Blue-cyan-yellow-red-darkred =>less to more correlated
                                              0 ---------------------> 1
                                              Expect a darkred line running from top left to bottom right
    """
    corr = df.corr() #data from corelation function
    fig,ax = plt.subplots(figsize = (size,size))
    ax.matshow(corr)  #color code for the rectangle by corelation values
    plt.xticks(range(len(corr.columns)),corr.columns) #draw x ticks marks
    plt.yticks(range(len(corr.columns)),corr.columns) #draw y ticks marks


# In[9]:


plot_corr(df)


# In[10]:


df.corr()


# In[11]:


df.head()


# In[12]:


del df['skin']


# In[13]:


df.head()


# # # Molding the data

# Check data types

# In[14]:


df.head(5)


# In[15]:


diabetes_map = {True:1,False:0}


# In[16]:


df['diabetes'] = df['diabetes'].map(diabetes_map)


# In[17]:


df.head()


# In[18]:


df.head()


# Now we are going to use this data to train the algorithms

# In[19]:


num_true= len(df.loc[df['diabetes']==True])
num_false= len(df.loc[df['diabetes']==False])
print(num_true)
print(num_false)
total = (num_true + num_false)*1.00
print total
per_true = 100*(num_true / total)
per_false = 100*(num_false / total)
print per_true
print per_false


# # #Splitting the data 70% for training , 30 % for testing

# In[20]:


#!/usr/bin/env python


# In[21]:


from sklearn.cross_validation import train_test_split


# In[22]:


#!/usr/bin/env python


# In[23]:


from sklearn.cross_validation import train_test_split


# In[24]:


from sklearn.cross_validation import train_test_split

feature_col_names = ['num_preg','glucose_conc','diastolic_bp','thickness','insulin','bmi','diab_pred',
                    'age']
predicted_class_names = ['diabetes']

X = df[feature_col_names].values
y = df[predicted_class_names].values
split_test_size = 0.30

X_train , X_test , y_train , y_test = train_test_split(X,y , test_size = split_test_size , random_state = 42)


# In[25]:


df.head()


# In[26]:


from sklearn.preprocessing import Imputer
# impute with mean all 0 reading
fill_0 = Imputer(missing_values=0, strategy = "mean" , axis =0)

X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)


# # Training Initial Algorithms - Naive Bayes

# In[27]:


from sklearn.naive_bayes import GaussianNB

#create Gaussian Naive Bayes model object and train it with the data 
nb_model = GaussianNB()

nb_model.fit(X_train,y_train.ravel())


# Performance on Training data

# In[28]:


#predict values using the training data

nb_predict_train = nb_model.predict(X_train)

#import the performance matrix library
from sklearn import metrics

#Accuracy 
print ("Accuracy : {0:.4f}".format(metrics.accuracy_score(y_train,nb_predict_train)))


# In[29]:


#predict values using the testing data
nb_predict_test = nb_model.predict(X_test)

from sklearn import metrics

#traning metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test , nb_predict_test)))


# In[30]:


df.head()


# # Metrics

# In[31]:


print("Confusion Matrix")
#Note the use of labels for set 1 = True to upper left and 0=False to lower right
print("{0}".format(metrics.confusion_matrix(y_test,nb_predict_test , labels = [1,0])))
print("")

print("Classification Report")
print(metrics.classification_report(y_test,nb_predict_test , labels = [1,0]))


# # Using Random Forest algorith 

# In[32]:


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state = 42)  #Create random forest object
rf_model.fit(X_train , y_train.ravel())


# # Predicting Training Data

# In[33]:


rf_predict_train = rf_model.predict(X_train)
#training metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train ,rf_predict_train)))


# # Predict Test Data

# In[34]:


rf_predict_test = rf_model.predict(X_test)

#training metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test , rf_predict_test)))


# # Metrix

# In[35]:


print("Confusion Matrix")
#Note the use of labels for set 1 = True to upper left and 0=False to lower right
print("{0}".format(metrics.confusion_matrix(y_test,nb_predict_test , labels = [1,0])))
print("")

print("Classification Report")
print(metrics.classification_report(y_test,nb_predict_test , labels = [1,0]))


# # Logistic Regression

# In[36]:


from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(C = 0.7 , random_state =42)
lr_model.fit(X_train , y_train.ravel())
lr_predict_test = lr_model.predict(X_test)


# In[37]:


#training metrics
print("Accuracy :{0:.4f}".format(metrics.accuracy_score(y_test , lr_predict_test)))
print(metrics.confusion_matrix(y_test , lr_predict_test , labels = [1,0]))
print("")

print("Classification Report")
print(metrics.classification_report(y_test , lr_predict_test , labels = [1,0]))


# # Setting regularization parameter
# 

# In[38]:


C_start = 0.1
C_end = 5
C_inc = 0.1

C_values , recall_scores = [] ,[]

C_val = C_start
best_recall_score = 0
while (C_val < C_end) :
    C_values.append(C_val)
    lr_model_loop = LogisticRegression(C = C_val, class_weight = "balanced" ,random_state = 42)
    lr_model_loop.fit(X_train , y_train.ravel())
    lr_predict_loop_test = lr_model_loop.predict(X_test)
    recall_score = metrics.recall_score(y_test , lr_predict_loop_test)
    recall_scores.append(recall_score)
    if (recall_score > best_recall_score):
        best_recall_score = recall_score
        best_lr_predict_test = lr_predict_loop_test
        
    C_val = C_val + C_inc
    
best_score_C_val = C_values[recall_scores.index(best_recall_score)]
print("1st max value of {0:.3f} occured at C = {1:.3f}".format(best_recall_score ,best_score_C_val ))

get_ipython().magic(u'matplotlib inline')
plt.plot(C_values , recall_scores, "-")
plt.xlabel("C value")
plt.ylabel("recall score")


# In[39]:


from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(class_weight = "balanced",C = best_score_C_val , random_state =42)
lr_model.fit(X_train , y_train.ravel())
lr_predict_test = lr_model.predict(X_test)

#training metrics
print("Accuracy :{0:.4f}".format(metrics.accuracy_score(y_test , lr_predict_test)))
print(metrics.confusion_matrix(y_test , lr_predict_test , labels = [1,0]))
print("")

print("Classification Report")
print(metrics.classification_report(y_test , lr_predict_test , labels = [1,0]))


# # LogisticRegressionCV

# In[41]:


from sklearn.linear_model import LogisticRegressionCV
lr_cv_model = LogisticRegressionCV(n_jobs = -1 , random_state = 42 , Cs = 3 , cv = 10 , refit =  True , class_weight = "balanced")
lr_cv_model.fit(X_train , y_train.ravel())


# In[42]:


lr_cv_predict_test = lr_cv_model.predict(X_test)
#training metrics
print("Accuracy :{0:.4f}".format(metrics.accuracy_score(y_test , lr_predict_test)))
print(metrics.confusion_matrix(y_test , lr_predict_test , labels = [1,0]))
print("")

print("Classification Report")
print(metrics.classification_report(y_test , lr_predict_test , labels = [1,0]))

