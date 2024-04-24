#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder


# In[2]:


df=pd.read_csv("apple_quality.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.nunique()


# In[7]:


df.duplicated().sum()


# In[8]:


msno.bar(df)
plt.show()


# In[9]:


df.dropna(inplace=True)


# In[10]:


df.drop(columns="A_id",inplace=True)


# In[11]:


df.tail()


# In[12]:


def inplace_encode_categorical_columns(df):

    label_encoders = {}
    for column in df.select_dtypes(include='object').columns:
        df[column] = df[column].astype(str)
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return label_encoders


# In[13]:


inplace_encode_categorical_columns(df)


# In[14]:


df.info()


# In[15]:


df.Quality.value_counts()


# # EDA

# In[16]:


sns.heatmap(df.corr(),cmap='magma',linewidths=5,square=True,annot=True)
plt.title("Visualize the Correlation Map ",color ="b")
plt.show()


# In[17]:


sns.histplot(x = 'Acidity' , data = df ,hue="Quality");


# In[18]:


sns.pairplot(df,hue='Quality',palette="mako")


# In[19]:


plt.figure(figsize=(15,10))
sns.set_palette('tab10')
for i,feature in enumerate(df.columns[:-1]):
    plt.subplot(3,3,i+1)
    sns.violinplot(x='Quality', y=feature, data=df)
    plt.title(f'{feature} distribution by Quality')
plt.tight_layout()
plt.show()


# In[20]:


plt.figure(figsize=(8, 8))
sns.set_palette('Set3')

for i, feature in enumerate(df.columns[:-1]):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(x='Quality', y=feature, data=df)
    plt.title(f'{feature} distribution by Quality')

plt.tight_layout()
plt.show()


# In[21]:


sns.countplot(x='Quality',data=df,palette="mako")


# In[22]:


df['Quality'].value_counts()


# In[23]:


df.dtypes


# 

# # Feature Scaling  
# 

# In[26]:


X = df.drop(['Quality'],axis = 1)
y = df['Quality']


# In[28]:


from sklearn.model_selection import train_test_split


# In[30]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=42)


# In[31]:


from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(X_train)    
x_test= st_x.transform(X_test)      


# ## Choose Best Model

# In[35]:


import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

best_model = None
best_accuracy = 0
best_difference = float('inf')

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'GaussianNB':GaussianNB()
    
}

# Fit and predict for each model
for name, model in models.items():
    # Fit the model
    model.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Predict on train data
    y_pred_train = model.predict(X_train)

    # Calculate accuracy
    accuracy_test = accuracy_score(y_test, y_pred)
    accuracy_train = accuracy_score(y_train, y_pred_train)

    # Print model scores
    print(f"Model: {name}")
    print(f"{name} Test Accuracy: {accuracy_test:.4f}")
    print(f"{name} Train Accuracy: {accuracy_train:.4f}")

    print("\nCompare the train-set and test-set accuracy\n")
    print("Check for overfitting and underfitting\n")
    print('Training set score: {:.4f}'.format(model.score(X_train, y_train)))
    print('Test set score: {:.4f}\n'.format(model.score(X_test, y_test)))    
    # Check for overfitting
    difference = abs(accuracy_train - accuracy_test)
    print(f"Difference between training and testing accuracy: {difference:.4f}")
    print(100*"*")

    # Update best model if it has the highest testing accuracy and minimal overfitting
    if accuracy_test > best_accuracy and difference < best_difference:
        best_model = model
        best_accuracy = accuracy_test
        best_difference = difference
        
   
print(f"Best Model: {best_model}")
print(f"Best Testing Accuracy: {best_accuracy:.4f}")
print(f"Difference between training and testing accuracy for the best model: {best_difference:.4f}")


# In[ ]:




