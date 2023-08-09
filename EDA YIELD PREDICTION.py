#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Necessary Imports
import os
import pandas as pd
import numpy as np
import datetime

import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train_data = pd.read_csv('train_data.csv')
train_weather = pd.read_csv('train_weather-1646897968670.csv')
farm_data = pd.read_csv('farm_data-1646897931981.csv')


# In[3]:


train_data.head()


# In[4]:


train_weather.head()


# In[5]:


train_weather.tail()


# In[6]:


farm_data.head()


# In[7]:


train_data.dtypes


# In[8]:


print(train_data.shape)


# In[9]:


train_weather.dtypes


# In[10]:


print(train_weather.shape)


# In[11]:


farm_data.dtypes


# In[12]:


# Distribution of the categorical variables levels .
train_data.describe(include = ['object'])


# In[13]:


train_weather.describe(include = ['object'])


# In[14]:


farm_data.describe(include = ['object'])


# In[15]:


# Distribution of the Numerical Columns.
train_weather.describe()


# In[16]:


farm_data.describe()


# In[17]:


#Train Dataset unique value count analysis.
train_data.nunique(axis = 0, dropna = False)


# In[18]:


# Graphical analysis of ingredient types. 
plt.figure(figsize=(8,6))
plt.title("Ingredients Types Frequency Value Count")
sns.countplot(x="ingredient_type", data= train_data)
plt.ylabel("Count")
plt.show()

#Unique value count of the ingredient_type column
train_data.ingredient_type.value_counts(normalize = True)*100


# In[19]:


## as we can see from above the ingredient (ing_w) has around 60% that means data is mostly on ingredient type ing_w


# In[20]:


train_data.farm_id.value_counts()


# In[23]:


# Farm Dataset unique value count analysis.
farm_data.nunique(axis = 0 , dropna = False)


# In[24]:



#Countplot to display the observations/max occurence in 'farming_company' column
plt.figure(figsize=(10,10))
sns.countplot(y="farming_company", data= farm_data )
plt.title("Maximum Occurence of each Farming Company")
plt.ylabel("Farming Companies")
plt.xlabel("Value Count")
plt.xticks(rotation=90)
plt.show()

#Unique value count of the farming_company column
farm_data.farming_company.value_counts()


# In[25]:


##Obery Farms                            549
#Wayne Farms                            279
#Sanderson Farms                        184
#Del Monte Foods                        156
#Dole Food Company                      147
##these are the top five companies from the above visualization


# In[26]:


#Count Plot to display the Unique locations
plt.figure(figsize=(22,8))
sns.countplot(x="deidentified_location", data= farm_data)
plt.title("THE UNIQUE LOCATIONS AT WHICH THE FARMS ARE PRESENT.")
plt.xticks(rotation = 45)
plt.show()

# Value count of unique locations
farm_data.deidentified_location.value_counts()


# In[27]:


#Plot for distribution of Farm Area.
plt.figure(figsize=(18,8))
sns.set(style="darkgrid")
sns.distplot(farm_data['farm_area'], kde = False)

plt.xlabel("Farm Area in Square Meters")
plt.xticks(rotation= 0)

plt.title('Farm Area Distribution')
plt.show()


# In[30]:


farm_data.num_processing_plants.nunique()


# In[28]:


farm_data.num_processing_plants.value_counts()


# In[31]:


# The Average Farm area of each company?
df = farm_data.groupby(['farming_company'])['farm_area'].mean().reset_index()
#df.reset_index()
df.sort_values('farm_area', ascending = False)


# In[32]:


#Bar Plot which depicts the avg farm area possessed
plt.figure(figsize = (28,7))
sns.barplot(x = 'farming_company', y = 'farm_area', data = farm_data)
plt.xlabel('FARMING COMPANIES')
plt.ylabel('AVG FARM AREA IN METRES')
plt.xticks(rotation= 45)

plt.title('AVERAGE FARM AREA OF FARMING COMPANIES ')
plt.show()


# In[33]:


# Which Farming Company Acquires more Farm Area ?
df1 = farm_data.groupby(['farming_company'])['farm_area'].sum().reset_index()
#df.reset_index()
df1.sort_values('farm_area', ascending = False)


# In[34]:


#The Avg Pressure Sea level and Avg Observed temp at different locations.
df3 = train_weather.groupby(['deidentified_location'])['pressure_sea_level','temp_obs'].mean().reset_index()
#df.reset_index()
df3.sort_values('pressure_sea_level', ascending = False)


# In[35]:


# Scatter plot to display the relation between the 'pressure_sea_level'  and 'temp_obs'
plt.figure(figsize=(10,10))
sns.set(style="darkgrid")
sns.scatterplot(x ='pressure_sea_level', y = 'temp_obs', data= train_weather)
plt.title("Sea Level Pressure Vs Observed Temperature")
plt.xlabel('Sea Level Pressure (Milli Bars)')
plt.ylabel('Observed Temp (Centigrade)')

plt.show()


# In[ ]:




