#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from datetime import datetime
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# In[4]:


# Reading the comma seperated value file by using pandas operation. 
df= pd.read_csv('D:\\mlproject\\bike project\\SeoulBikeData.csv.xls', encoding= 'unicode_escape')


# In[5]:


df


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


# Here we describe the dataset which shows aggregated and percentagewise values.

df.describe().T


# In[9]:


# Mentioned the shape of data.

print(f' We have total {df.shape[0]} rows and {df.shape[1]} columns.')


# In[10]:


# Important columns names  which is included in dataset.

column=df.columns
column
     


# In[11]:


# Checking the null values of data.

df.isnull().sum()


# In[12]:


# Checking missing values using heatmap.

plt.figure(figsize=(14, 5))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False)
plt.xlabel("column_name", size=14, weight="bold")
plt.title("missing values in column",fontweight="bold",size=17)
plt.show()


# In[13]:


# Checking Duplicate Values

value=len(df[df.duplicated()])
print("The number of duplicate values in the data set is = ",value)


# In[14]:


# Extracting categorical features from dataset.

categorical_features= df.select_dtypes(include='object')
categorical_features


# In[15]:


# Calculating value of counts of some features of dataset.

df['Functioning Day'].value_counts()


# In[16]:


df['Holiday'].value_counts()


# In[17]:


df['Seasons'].value_counts()


# In[18]:


# Converting string format of 'Date' column into date-time format.

df['Date'] = pd.to_datetime(df['Date'])


# In[19]:


# Creating a column containing the year from a particular date.

year = []
for i in range(len(df['Date'])):
  year.append(df['Date'][i].year)
df['year'] = year


# In[20]:


# Creating a column containing the month number from a particular date.

months = []
for i in range(len(df['Date'])):
  months.append(df['Date'][i].month)
df['month'] = months


# In[21]:


# Group by the data on the basis of rented bike count.

df_Date = df.groupby('Date').sum()['Rented Bike Count']
df_Date


# In[22]:


# Creating a series which shows total number of bikes rented on the type of day

df_holiday = df.groupby('Holiday').sum()['Rented Bike Count']
df_holiday


# In[23]:


# Group by the data on the basis of seasons. 

season_by= df.groupby('Seasons').sum()
season_by


# In[24]:


# Graphical representation of rented bike count along with seasons. 

season_by['Rented Bike Count'].plot(kind='pie', subplots=True, autopct='%1.1f%%', figsize=(5, 5));
     


# In[25]:


# Differentiation between holiday and working day on seasonal basis. 

plt.figure(figsize=(10,7))
sns.countplot(x=df['Seasons'],hue=df['Holiday'])
plt.title ("season wise booking")
plt.xlabel('seasons')
plt.ylabel('count');


# In[26]:


# Checking correlation using heatmap

plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),cmap='PiYG',annot=True);


# In[27]:


# Plotting cat plot for more info with season wise.
 
sns.catplot(x='Seasons',y='Rented Bike Count',data=df);


# In[28]:


# On hourly basis rented bike count.

hour_wise = df.groupby('Hour').sum()['Rented Bike Count']
hour_wise


# In[29]:


# Rented bike count as per hour,holidays,seasons,functional day.

fig, axs = plt.subplots(nrows=4,ncols=1,figsize=(12,10), dpi=100)
sns.pointplot(data=df, x="Hour", y="Rented Bike Count", ax=axs[0])
sns.pointplot(data=df, x="Hour", y="Rented Bike Count", ax=axs[1], 
              hue="Holiday")
sns.pointplot(data=df, x="Hour", y="Rented Bike Count", ax=axs[2], 
              hue="Functioning Day")
sns.pointplot(data=df, x="Hour", y="Rented Bike Count", ax=axs[3], 
              hue="Seasons")
plt.tight_layout()
     


# In[30]:


# Plotting the independent variables on the basis of dependent variables.
  
for col in column[1:]:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    feature = df[col]
    feature.hist(bins=50, ax = ax)   
    ax.set_title(col)
plt.show()


# In[31]:


# Creating a dataframe containing the count of bikes rented in different intensities of rainfall.

df_temp = pd.DataFrame(df.groupby('Temperature(°C)')['Rented Bike Count'].sum())
df_temp.reset_index(inplace=True)
df_temp.head()


# In[32]:


# Creating a dummy variables for the season column.

df['Winter'] = np.where(df['Seasons']=='Winter', 1, 0)
df['Spring'] = np.where(df['Seasons']=='Spring', 1, 0)
df['Summer'] = np.where(df['Seasons']=='Summer', 1, 0)
df['Autumn'] = np.where(df['Seasons']=='Autumn', 1, 0)

df.drop(columns=['Seasons'],axis=1,inplace=True)


# In[33]:


# encoding 'Holiday' column with 0 and 1.

for i in range(len(df['Holiday'])):
  if df['Holiday'][i] == 'No Holiday':
    df['Holiday'][i] = 0
  else:
    df['Holiday'][i] = 1 


# In[34]:


# encoding 'Functioning Day' column with 0 and 1.

for i in range(len(df['Functioning Day'])):
  if df['Functioning Day'][i] == 'Yes':
    df['Functioning Day'][i] = 1
  else:
    df['Functioning Day'][i] = 0


# In[35]:


# Taking the look for the data whether the operation is placed or not.
 
df.head()
     


# In[36]:


# scatter plot of bike count on differant dates.

plt.figure(figsize=(10,6))
plt.scatter(df['Date'], df['Rented Bike Count'], alpha=0.5)
plt.title('Scatter plot of bike count with dates')
plt.xlabel('Date')
plt.ylabel('Bike count')
plt.show()


# In[37]:


# finding the inter-quartile range.
 
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[38]:


# listing features to remove outliers.

features = list(df.columns)
features = features[2:]
list_0 = ['Hour','Winter','Spring','Summer','Autumn','Holiday','Functioning Day','month','year']
new_features = [x for x in features if x not in list_0]


# In[39]:


# printing the column for new features.

new_features


# In[40]:


# Finding the lower and the upper range of the outliers.

df[new_features] = df[new_features][~((df[new_features] < (Q1 - 1.5 * IQR)) |
                                      (df[new_features] > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[41]:


# checking the null value count after removing the outliers.

df.info()


# In[42]:


# filling null values with mean values.

df['Temperature(°C)'] = df['Temperature(°C)'].fillna(df['Temperature(°C)'].mean())

df['Humidity(%)'] = df['Humidity(%)'].fillna(df['Humidity(%)'].mean())

df['Wind speed (m/s)'] = df['Wind speed (m/s)'].fillna(df['Wind speed (m/s)'].mean())

df['Visibility (10m)'] = df['Visibility (10m)'].fillna(df['Visibility (10m)'].mean())

df['Dew point temperature(°C)'] = df['Dew point temperature(°C)'].fillna(df['Dew point temperature(°C)'].mean())

df['Solar Radiation (MJ/m2)'] = df['Solar Radiation (MJ/m2)'].fillna(df['Solar Radiation (MJ/m2)'].mean()) 

df['Rainfall(mm)'] = df['Rainfall(mm)'].fillna(df['Rainfall(mm)'].mean())

df['Snowfall (cm)'] = df['Snowfall (cm)'].fillna(df['Snowfall (cm)'].mean())


# In[43]:


# checking correlation from heatmap.

plt.figure(figsize=(15,12))
sns.heatmap(df.corr('pearson'),vmin=-1, vmax=1,cmap='coolwarm',annot=True, square=True);


# In[44]:


# dropping columns with more (or less) correlation.

df.drop(columns=['Dew point temperature(°C)','Date','Rainfall(mm)','Snowfall (cm)','year'],axis=1,inplace=True)


# In[45]:


# extracting correlation heatmap.

plt.figure(figsize=(15,12))
sns.heatmap(df.corr('pearson'),annot=True, square=True);


# In[46]:


# function to calculate Multicollinearity.

from statsmodels.stats.outliers_influence import variance_inflation_factor
def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)
     


# In[47]:


# multicollinearity result.

calc_vif(df[[i for i in df.describe().columns if i not in ['Rented Bike Count','Date']]])


# In[48]:


# dropping "summer" column as it adds to multicollinearity.

df.drop(columns=['Summer'],axis=1,inplace=True)


# In[49]:


calc_vif(df[[i for i in df.describe().columns if i not in ['Rented Bike Count','Date']]])


# In[50]:


# Taking a look at the data after dealing with outliers (filling the values with mean).

df.info() 


# In[51]:


# converting object type columns to float.

df['Functioning Day'] = df['Functioning Day'].astype(float)
df['Holiday'] = df['Holiday'].astype(float)


# In[52]:


# obtaining correlation plots between dependent and independent variables.

numeric_features = df.columns
for col in numeric_features[1:]:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    feature = df[col]
    label = df['Rented Bike Count']
    correlation = feature.corr(label)
    plt.scatter(x=feature, y=label)
    plt.xlabel(col)
    plt.ylabel('Rented Bike Count')
    ax.set_title('Rented Bike Count vs ' + col + '- correlation: ' + str(correlation))
    z = np.polyfit(df[col], df['Rented Bike Count'], 1)
    y_hat = np.poly1d(z)(df[col])

    plt.plot(df[col], y_hat, "r--", lw=1)

plt.show()


# In[55]:


X = df.drop(columns=['Rented Bike Count'],axis=1).values
X


# In[60]:


y = df['Rented Bike Count']
y


# In[61]:


##Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[62]:


X_train


# In[63]:


X_test


# In[114]:


## Standardize the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[115]:


X_train = scaler.fit_transform(X_train)


# In[116]:


X_test = scaler.transform(X_test)
     


# In[117]:


import pickle
pickle.dump(scaler,open('scaling.pkl','wb'))


# In[118]:


X_train


# In[119]:


X_test


# In[120]:


# Model Training.
from sklearn.linear_model import LinearRegression


# In[121]:


regression=LinearRegression()


# In[122]:


regression.fit(X_train,y_train)


# In[123]:


## print the coefficients and the intercept
print(regression.coef_)


# In[124]:


print(regression.intercept_)


# In[125]:


## on which parameters the model has been trained
regression.get_params()


# In[126]:


### Prediction With Test Data
reg_pred=regression.predict(X_test)


# In[127]:


reg_pred


# In[128]:


## plot a scatter plot for the prediction
plt.scatter(y_test,reg_pred)


# In[129]:


## Residuals
residuals=y_test-reg_pred


# In[130]:


residuals


# In[131]:


## Plot this residuals 

sns.displot(residuals,kind="kde")


# In[132]:


## Scatter plot with respect to prediction and residuals
## uniform distribution
plt.scatter(reg_pred,residuals)


# In[133]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test,reg_pred))
print(mean_squared_error(y_test,reg_pred))
print(np.sqrt(mean_squared_error(y_test,reg_pred)))


# In[134]:


from sklearn.metrics import r2_score
score=r2_score(y_test,reg_pred)
print(score)


# In[135]:


#display adjusted R-squared
1 - (1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)


# In[143]:


df.shape


# In[147]:


df.columns


# In[158]:


import pickle


# In[159]:


pickle.dump(regression,open('regmodel.pkl','wb'))


# In[160]:


pickled_model=pickle.load(open('regmodel.pkl','rb'))


# In[ ]:




