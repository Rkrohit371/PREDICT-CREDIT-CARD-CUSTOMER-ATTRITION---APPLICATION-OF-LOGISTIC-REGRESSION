#!/usr/bin/env python
# coding: utf-8

# ## Import libraries and data

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


data = pd.read_csv('Banking_CreditCardAttrition.csv', sep=';')
data.head()


# ## Data Visualization and Preprocessing

# In[5]:


print(data.info())


# In[42]:


print(sum(data.duplicated()))


# In[6]:


data.describe()


# In[7]:


data.isnull().sum()


# In[94]:


data['Customer_Age'].describe()


# In[97]:


data['Credit_Limit'].describe()


# In[8]:


data['Education_Level'].value_counts()


# In[9]:


data['Marital_Status'].value_counts()


# In[10]:


data['Income_Category'].value_counts()


# In[11]:


data['Card_Category'].value_counts()


# In[12]:


data['Gender'].value_counts()


# In[79]:


sns.set_style("whitegrid")
data['Customer_Age'].hist(bins=40, figsize=(10, 8))


# In[83]:


sns.countplot(x='Attrition_Flag', data=data, hue='Gender')


# In[88]:


plt.figure(figsize=(7, 7))
sns.countplot(x='Attrition_Flag', data=data, hue='Marital_Status')


# In[90]:


plt.figure(figsize=(7, 7))
sns.countplot(x='Attrition_Flag', data=data, hue='Education_Level')


# In[91]:


plt.figure(figsize=(7, 7))
sns.countplot(x='Attrition_Flag', data=data, hue='Income_Category')


# In[92]:


plt.figure(figsize=(7, 7))
sns.countplot(x='Attrition_Flag', data=data, hue='Card_Category')


# In[98]:





# In[ ]:





# In[19]:


data.corr()


# In[24]:


plt.figure(figsize=(2, 2))
corr = data.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# In[44]:


def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))

Customer_Age_outlier = outliers_iqr(data['Customer_Age'])
Customer_Age_outlier = Customer_Age_outlier[0]


# In[47]:


Dependent_count_outlier = outliers_iqr(data['Dependent_count'])
Dependent_count_outlier = Dependent_count_outlier[0]
Dependent_count_outlier


# In[50]:


Months_on_book_outlier = outliers_iqr(data['Months_on_book'])
Months_on_book_outlier = Months_on_book_outlier[0]
Months_on_book_outlier.shape


# In[51]:


Total_Relationship_Count_outlier = outliers_iqr(data['Total_Relationship_Count'])
Total_Relationship_Count_outlier = Total_Relationship_Count_outlier[0]
Total_Relationship_Count_outlier.shape


# In[54]:


Months_Inactive_12_mon_outlier = outliers_iqr(data['Months_Inactive_12_mon'])
Months_Inactive_12_mon_outlier = Months_Inactive_12_mon_outlier[0]
Months_Inactive_12_mon_outlier.shape


# In[55]:


Contacts_Count_12_mon_outlier = outliers_iqr(data['Contacts_Count_12_mon'])
Contacts_Count_12_mon_outlier = Contacts_Count_12_mon_outlier[0]
Contacts_Count_12_mon_outlier.shape


# In[56]:


Credit_Limit_outlier = outliers_iqr(data['Credit_Limit'])
Credit_Limit_outlier = Credit_Limit_outlier[0]
Credit_Limit_outlier.shape


# In[57]:


Total_Revolving_Bal_outlier = outliers_iqr(data['Total_Revolving_Bal'])
Total_Revolving_Bal_outlier = Total_Revolving_Bal_outlier[0]
Total_Revolving_Bal_outlier.shape


# In[62]:


Total_Revolving_Bal_outlier


# In[61]:


Total_Revolving_Bal_outlier


# In[66]:


sns.boxplot(data['Months_on_book'])


# In[67]:


sns.boxplot(data['Months_Inactive_12_mon'])


# In[68]:


sns.boxplot(data['Contacts_Count_12_mon'])


# In[69]:


sns.boxplot(data['Credit_Limit'])


# In[75]:


plt.figure(figsize =(10, 7))
columns = ['Contacts_Count_12_mon', 'Months_Inactive_12_mon', 'Months_on_book']
df = pd.DataFrame(data = data, columns = columns)
sns.boxplot(x="variable", y="value", data=pd.melt(df))
plt.show()


# In[76]:


floor = data["Customer_Age"].quantile(0.05)
cap = data["Customer_Age"].quantile(0.99)
print(floor)
print(cap)


# In[99]:


data['Customer_Age'].fillna(46.32, inplace=True)
data['Credit_Limit'].fillna(9404.8, inplace=True)


# In[100]:


data.info()


# In[110]:


gender_dm = pd.get_dummies(data['Gender'], drop_first=True)
gender_dm.columns = ['Male']
gender_dm.head()


# In[111]:


Education_Level_dm = pd.get_dummies(data['Education_Level'], drop_first=True)
Education_Level_dm.head()


# In[112]:


Marital_Status_dm = pd.get_dummies(data['Marital_Status'], drop_first=True)
Marital_Status_dm.head()


# In[113]:


Income_Category_dm = pd.get_dummies(data['Income_Category'], drop_first=True)
Income_Category_dm.head()


# In[114]:


Card_Category_dm = pd.get_dummies(data['Card_Category'], drop_first=True)
Card_Category_dm.head()


# In[121]:


new_data = pd.concat([data, gender_dm, Education_Level_dm, Card_Category_dm, Income_Category_dm, Marital_Status_dm], axis=1)
new_data.head()


# In[122]:


new_data.columns


# In[123]:


new_data.drop(['CLIENTNUM', 'Education_Level', 'Marital_Status', 'Card_Category', 'Gender', 'Income_Category'], inplace=True, axis=1)
new_data.head()


# In[135]:


plt.figure(figsize=(10, 10))
sns.heatmap(new_data.corr(), cmap='coolwarm')


# In[141]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# In[145]:


ck = add_constant(new_data)
pd.Series([variance_inflation_factor(ck.values, i) 
               for i in range(ck.shape[1])], 
              index=ck.columns)


# ## Training and Evaluation

# ### Logistic Regression Classifier

# In[182]:


X = new_data.drop('Attrition_Flag', axis=1)
# X = new_data[columns]
y = new_data['Attrition_Flag']


# In[183]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)


# In[184]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[185]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[186]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[187]:


lr.fit(X_train, y_train)


# In[188]:


preds = lr.predict(X_test)
preds


# In[189]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, preds))


# In[190]:


print(confusion_matrix(y_test, preds))


# In[ ]:





# In[173]:


columns = new_data.columns[9:21]

