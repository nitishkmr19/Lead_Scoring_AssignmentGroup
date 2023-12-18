#!/usr/bin/env python
# coding: utf-8

# # Lead Scoring Assignment

# ### Problem Statement
# An education company named X Education sells online courses to industry professionals. On any given day, many professionals who are interested in the courses land on their website and browse for courses. 
# 
#  X Education has appointed you to help them select the most promising leads, i.e. the leads that are most likely to convert into paying customers. The company requires you to build a model wherein you need to assign a lead score to each of the leads such that the customers with a higher lead score have a higher conversion chance and the customers with a lower lead score have a lower conversion chance. The CEO, in particular, has given a ballpark of the target lead conversion rate to be around 80%.

# Data - Leads.csv

# In[1]:


#Import the packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Import the data:
df_leads= pd.read_csv('Leads.csv')


# In[4]:


#Read the data
df_leads.head()


# In[5]:


# Inspect the data
df_leads.describe()


# In[6]:


df_leads.info()


# In[7]:


df_leads.shape


# In[8]:


# Checking for duplicates
df_leads.loc[df_leads.duplicated()]


# ##### No duplicates in the data

# In[9]:


df_leads.dtypes


# Total 7 numeric columns and 30 categorical columns.

# #### some categorical variables have a level called 'Select' which needs to be handled.

# In[10]:


df_leads = df_leads.replace('Select', np.nan)


# In[11]:


df_leads.isnull().sum()


# In[12]:


#Checking percentage of missing/null values
round(100*(df_leads.isnull().sum()/len(df_leads)),2)


# Note:Lead Quality, Tags, Asymmetrique scores ,Profile and Last Notable Activity are created by the sales team after contacting the leads so we can drop these columns.

# In[13]:


# Listing all the columns having more than 30% missing values into 'missing_columns':

missing_columns_30 = df_leads.columns[100*(df_leads.isnull().sum()/len(df_leads)) > 30]
print(missing_columns_30)


# In[14]:


miss_col=missing_columns_30.drop('Specialization')


# Note - Creating a new dataframe 'lead_df1' as copy of original dataframe 'df_leads' so that while dropping missing value columns our original dataframe remains unaffected

# In[15]:


# Creating copy of original datarframe 

lead_df1=df_leads.copy()


# In[16]:


# Droping the columns having more than 30% missing values

lead_df1 = lead_df1.drop(miss_col, axis=1).copy()


# In[17]:


lead_df1.shape


# In[18]:


# Checking the remaining columns for missing values

round(100*(lead_df1.isnull().sum()/len(lead_df1)),2)


# In[22]:


#Set_option to avoid truncation of columns and rows:-

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[24]:


# Plotting count plot of 'Specialization' to see the data distribution:-

plt.figure(figsize=(15,5))
sns.countplot(x='Specialization', data = lead_df1)
plt.xticks(rotation=90)
plt.show()


# In[25]:


# Imputing missing value with 'Other' 

lead_df1['Specialization'].fillna('Other',inplace=True)


# In[26]:


# Listing all the columns having missing values into 'remaining_missing'

remaining_missing = lead_df1.columns[(100*(lead_df1.isnull().sum()/len(lead_df1)) < 30) & (100*(lead_df1.isnull().sum()/len(lead_df1)) >0) ]
print(remaining_missing)


# In[27]:


lead_df1['What is your current occupation'].value_counts()


# In[31]:


plt.figure(figsize=(15,5))
sns.countplot(x='What is your current occupation', data = lead_df1)
plt.show()


#  'Unemployed' count is highest but we will impute missing values with 'Other' considering we do not know current occupation of lead.So we will create a separate category called 'Other'.

# In[32]:


lead_df1['What is your current occupation'].fillna('Other',inplace=True)


# In[34]:


plt.figure(figsize=(15,5))
sns.countplot(x='What is your current occupation', data = lead_df1)
plt.show()


# In[36]:


# Plotting count plot to visualize counts of data of 'Country' column

plt.figure(figsize=(15,5))
sns.countplot(x= 'Country', data = lead_df1)
plt.xticks(rotation=90)
plt.show()


# Note - From count plot we can see that 'India' count is highest so we can impute missing values with 'India'

# In[37]:


lead_df1.Country.fillna('India',inplace=True)


# In[38]:


lead_df1.Country.value_counts()


# In[39]:


# Dropping 'Country' column from dataframe

lead_df1.drop('Country',1,inplace=True)


# In[40]:


# Checking value counts of 'Lead Source'

lead_df1['Lead Source'].value_counts()


# In[41]:


# Replace 'google' with 'Google' 

lead_df1['Lead Source']=lead_df1['Lead Source'].replace('google','Google')


# In[42]:


lead_df1['Lead Source'].value_counts()


# In[43]:


# Imputing missing values with 'Google'

lead_df1['Lead Source'].fillna('Google',inplace=True)


# In[44]:


# Checking value counts of 'Last Activity':-

lead_df1['Last Activity'].value_counts()


# As we do not know the last activity of leads which are missing values and most frequent value is 'Email Opened' so we can impute missing value with 'Email Opened'.

# In[45]:


# Imputing missing values with 'Email Opened'

lead_df1['Last Activity'] = lead_df1['Last Activity'].replace(np.nan, 'Email Opened')


# In[46]:


# Imputing missing value with 'median' value for both 'TotalVisits' and 'Page Views Per Visit' columns

lead_df1['TotalVisits'].fillna(lead_df1['TotalVisits'].median(), inplace=True)

lead_df1['Page Views Per Visit'].fillna(lead_df1['Page Views Per Visit'].median(), inplace=True)


# In[47]:


lead_df1.shape


# In[48]:


# Checking missing values after treating missing values

round(100*(lead_df1.isnull().sum()/len(lead_df1)),2)


# In[49]:


# Dropping 'Prospect ID' and 'Lead Number' variables

lead_df1.drop(['Prospect ID','Lead Number'],1,inplace=True)


# In[50]:


# Dropping variables which are having imbalanced data:-

lead_df1.drop(['Do Not Call','What matters most to you in choosing a course','Search','Magazine','Newspaper Article','X Education Forums','Newspaper',
          'Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses','Update me on Supply Chain Content','Get updates on DM Content','I agree to pay the amount through cheque'],1,inplace=True)


# In[51]:


# Dropping column 'last Notable Activity'

lead_df1.drop('Last Notable Activity',1,inplace=True)


# In[52]:


lead_df1.shape


# In[53]:


lead_df1.head()


# ### Data Visualization

# In[54]:


# Calculating conversion rate:-

Converted = (sum(lead_df1['Converted'])/len(lead_df1['Converted'].index))*100
Converted


# In[56]:


#Checking value count of target variable 'Converted'

lead_df1.Converted.value_counts()


# In[59]:


#Plotting count plot to get clear view of data distribution of 'Converted' column:-

sns.countplot(x= 'Converted', data = lead_df1)
plt.title("Distribution of Converted Variable")
plt.show()


# In[60]:


# Plotting count plot of 'Lead Origin' for both 'Converted' 0 and 1 :-

plt.figure(figsize = (15,5))

ax=sns.countplot(x = "Lead Origin", hue = "Converted", data = lead_df1)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel=None)

plt.xticks(rotation = 90)
plt.title('Lead Origin',fontsize=20)
ax.set_yscale('log')

plt.show()


# Note:
# - Conversion rate for 'API' is ~ 31% and for 'Landing Page Submission' is ~36%.
# - For 'Lead Add Form' number of conversion is more than unsuccessful conversion.
# - Count of 'Lead Import' is lesser.

# In[63]:


# Plotting count plot of 'Lead Source' based on 'Converted' value 0 and 1 :-

plt.figure(figsize = (25,5))

ax=sns.countplot(x = "Lead Source", hue = "Converted", data = lead_df1)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel=None)

plt.xticks(rotation = 90)
plt.title('Lead Source',fontsize=20)

ax.set_yscale('log')# Setting logrithmic scale

plt.show()


# In[65]:


#Club the values
# Combining all low frequency values together:-

lead_df1['Lead Source'] = lead_df1['Lead Source'].replace(['blog','Pay per Click Ads','bing','Social Media','WeLearn','Click2call', 'Live Chat','welearnblog_Home', 'youtubechannel','testone',
                                                           'Press_Release','NC_EDM'], 'Others')


# In[66]:


# Again plotting count plot of 'Lead Source' based on 'Converted' value 0 and 1 :-

plt.figure(figsize = (25,5))

ax=sns.countplot(x = "Lead Source", hue = "Converted", data = lead_df1)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel=None)

plt.xticks(rotation = 90)
plt.title('Lead Source',fontsize=20)

ax.set_yscale('log')

plt.show()


# Note:
# - Google and Direct traffic generates maximum number of leads.
# - Conversion rate of 'Reference' and 'Welingak Website' leads is high.

# In[68]:


# Plotting count plot of 'Lead Source' based on 'Converted' value 0 and 1 :-

plt.figure(figsize = (20,5))

ax=sns.countplot(x = "Do Not Email", hue = "Converted", data = lead_df1)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel=None)

plt.xticks(rotation = 90)
plt.title('Do Not Email',fontsize=20)

ax.set_yscale('log')# Setting logrithmic scale

plt.show()


# Note:
# - People who opted for mail option are becoming more leads

# In[69]:


# Plotting count plot of 'Lead Source' based on 'Converted' value 0 and 1 

plt.figure(figsize = (20,5))

ax=sns.countplot(x = "Last Activity", hue = "Converted", data = lead_df1)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel=None)

plt.xticks(rotation = 90)
plt.title('Last Activity',fontsize=20)

ax.set_yscale('log')# Setting logrithmic scale

plt.show()


# In[70]:


# Combining all low frequency values together under label 'Others':-

lead_df1['Last Activity'] = lead_df1['Last Activity'].replace(['Had a Phone Conversation','View in browser link Clicked','Visited Booth in Tradeshow',
      'Approached upfront','Resubscribed to emails','Email Received','Email Marked Spam'],'Others')


# In[71]:


# Again plotting count plot of 'Last Activity' based on 'Converted' value 0 and 1 :-

plt.figure(figsize = (25,5))

ax=sns.countplot(x = "Last Activity", hue = "Converted", data = lead_df1)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
ax.set(xlabel=None)

plt.xticks(rotation = 90)
plt.title('Last Activity of Lead',fontsize=20)

ax.set_yscale('log')

plt.show()


# Note:
# - Conversion rate for last activity of 'SMS Sent'is ~63%.
# - Highest last activity of leads is 'Email Opened' .

# ### Outlier Analysis

# In[72]:


# Plotting box plots to visualize data distribution of 'TotalVisits':-

plt.figure(figsize = (7,4))
sns.boxplot(lead_df1['TotalVisits'],orient='v',palette='Set2')

plt.show()


# From above box plot that only upper range outliers are present in data, so need to treat outliers.

# In[73]:


# Treating outliers by capping upper range to 0.99:-

Q3 = lead_df1.TotalVisits.quantile(0.99)

lead_df1 = lead_df1[(lead_df1.TotalVisits <= Q3)]


# In[74]:


# Verifying outliers after removing it :-

plt.figure(figsize = (7,4))
sns.boxplot(y=lead_df1['TotalVisits'],palette='Set2')
plt.show()


# In[75]:


# Plotting box plots to visualize data distribution of 'Total Time Spent on Website':-

plt.figure(figsize=(7,4))
sns.boxplot(y=lead_df1['Total Time Spent on Website'],orient='v',palette='Set2')
plt.show()


# There is no outlier in data,so no trreatment required for it.

# ### Bivariate Analysis

# In[78]:


# Heatmap to understand the attributes correlation:-

plt.figure(figsize = (15,10))        
ax = sns.heatmap(lead_df1.corr(),annot = True,cmap='Reds')


# Note:
# - 'TotalVisits' and 'Page Views per Visit' are highly correlated with correlation of .72
# - 'Total Time Spent on Website' has correlation of 0.36 with target variable 'Converted'.

# In[79]:


# Plotting box plot of "Total Time Spent on Website" vs Converted variable to check data distribution:-

plt.figure(figsize=(10,5))
sns.boxplot(x='Converted', y='Total Time Spent on Website',data=lead_df1)
plt.show()


# Leads spending more time on website are more likely to opt for curses or converted.

# ### Converting some binary variables (Yes/No) to 0/1

# In[80]:


# Variable to map:-

var =  ['Do Not Email','A free copy of Mastering The Interview']

# Defining the map function
def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

# Applying the function to the variable:-

lead_df1[var] = lead_df1[var].apply(binary_map)


# In[81]:


lead_df1.head()


# #### Creating Dummy Variable

# In[82]:


# Creating list 'cat_col' of categorical columns :-

cat_col= lead_df1.select_dtypes(include=['object']).columns
cat_col


# In[83]:


# Creating dummy variable for some of the categorical variables and dropping the first one using 'drop_first=True':-

dummy = pd.get_dummies(lead_df1[['Lead Origin', 'Lead Source', 'Last Activity','What is your current occupation',
                             'Specialization']], drop_first=True)

dummy.head()


# In[84]:


# Adding dummy variables dataset 'dummy' to original dataset 'lead_df1':-

lead_df1= pd.concat([dummy,lead_df1],axis = 1)


# In[85]:


lead_df1.head()


# In[86]:


# Dropping repeated columns for which dummy variables were created:-

lead_df1.drop(['Lead Origin', 'Lead Source', 'Last Activity','What is your current occupation','Specialization'
                             ],1,inplace = True)


# In[87]:


lead_df1.shape


# In[88]:


lead_df1.info()


# ## Test-Train Split

# In[89]:


# Importing required library to split data

from sklearn.model_selection import train_test_split


# In[90]:


# Putting feature variable to X:-

X = lead_df1.drop(['Converted'], axis=1)

# Displaying head :-

X.head()


# In[91]:


# Putting response variable to y:-

y = lead_df1['Converted']

y.head()


# In[92]:


# Splitting the data into train and test of 70:30 ratio:-

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[93]:


# Checking shape of 'X_train' dataset after splitting:-

X_train.shape


# In[94]:


# Checking shape of 'X_test' dataset after splitting:-

X_test.shape


# In[95]:


# Verifying info of data set after splitting:-

lead_df1.info()


# ## Feature Scaling

# In[96]:


##Importing required library for scaling :-

from sklearn.preprocessing import StandardScaler


# In[97]:


# Creating 'scaler' object for 'StandardScaler':-

scaler = StandardScaler()

# Applying 'fit_transform' to scale the 'train' data set:-

X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])

# Displaying the head of the data after scaling:-
X_train.head()


# In[98]:


# Checking the Correlation Matrix

plt.figure(figsize = (55, 35),dpi=80)
sns.heatmap(lead_df1.corr(), annot = True, cmap="YlGnBu",annot_kws={"size": 18})
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.tight_layout()
plt.show()


# Note:
# 
# - The heatmap clearly shows which all variable are multicollinear in nature, and which variable have high collinearity with the target variable
# 
# - 'Lead Source_Facebook' and 'Lead Origin_Lead Import' having higher correlation of 0.98.
# - 'Do Not Email' and 'Last Activity_Email Bounced' having higher correlation.
# - 'Lead Origin_Lead Add Form' and 'Lead Source_Referance' having higher correlation of 0.85.
# - 'TotalVisits' and 'Page Views Per Visit' having correlation of 0.72.

# ## Model Building

# In[114]:


#Importing 'LogisticRegression' :-

from sklearn.linear_model import LogisticRegression

# Creating LogisticRegression Object called 'regressor':-

regressor = LogisticRegression()


# In[115]:


#Importing 'RFE' for feature selection:-

from sklearn.feature_selection import RFE


# In[116]:


rfe = RFE(regressor, n_features_to_select=15)

rfe = rfe.fit(X_train, y_train)


# In[117]:


#Displaying columns selected by RFE and their weights:-

list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[118]:


# Storing columns selected by RFE into 'col' and displaying it:-

col = X_train.columns[rfe.support_]
col


# In[119]:


# Displaying columns which are not selected by RFE:-

X_train.columns[~rfe.support_]


# In[120]:


# Creating X_test dataframe with RFE selected variables:-

X_train_rfe = X_train[col]


# In[121]:


X_train_rfe.head()


# ## Model 1

# In[122]:


#Importing required 'statsmodels' library:-

import statsmodels.api as sm


# In[123]:


# Add a constant:-

X_train_sm = sm.add_constant(X_train_rfe)

# Building first fitted model:-

logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial()).fit()


# In[124]:


# Summary of Logistic regression model :-
logm1.summary()


# ### VIF Check for multicollinearity : variance_inflation_factor

# In[125]:


# Importing 'variance_inflation_factor' from 'statsmodels':-

from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[126]:


# Creating dataframe called 'vif' containing names feature variables and their respective VIFs:-

vif = pd.DataFrame()
vif['Features'] = X_train_rfe.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe.values, i) for i in range(X_train_rfe.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[127]:


#Dropping 'What is your current occupation_Housewife' because of insignificant variable p-value=0.999(p>0.05):-

X_train_new = X_train_rfe.drop(["What is your current occupation_Housewife"], axis = 1)


# ## Model 2

# In[128]:


# Add a constant:-

X_train_sm2 = sm.add_constant(X_train_new)

# Building second fitted model:-

logm2 = sm.GLM(y_train,X_train_sm2, family = sm.families.Binomial()).fit()


# In[129]:


# Summary of the logistic regression model obtained:-

logm2.summary()


# In[130]:


# Calculating VIF for new model:-

vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[131]:


#Dropping 'Last Activity_Others' because of  p-value=0.01:-

X_train_new = X_train_new.drop(["Last Activity_Others"], axis = 1)


# ## Model 3

# In[132]:


#Adding constant:-

X_train_sm3 = sm.add_constant(X_train_new)

# Create a third fitted model:-

logm3 = sm.GLM(y_train,X_train_sm3, family = sm.families.Binomial()).fit()


# In[133]:


# Summary of the logistic regression model obtained:-

logm3.summary()


# In[134]:


# Calculating VIF for new model:-

vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Note:
# - From model 'logm3' we can see that P-values of variables are significant and VIF values are below 3 . So we need not drop any more variables and we can proceed with making predictions using this model only considering model 'logm3' as final model.

# ### Making prediction on 'train' dataset based on final model

# In[135]:


# Calculating predicted values of 'y_train':-

y_train_pred = logm3.predict(X_train_sm3)
                            
y_train_pred[:10] # Displaying 10 values


# In[136]:


# Reshaping :-

y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[137]:


# Creating dataframe 'y_train_pred_final' with actual and predicted :-

y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_Prob':y_train_pred})

# Adding column 'Prospect ID' for indexing:-

y_train_pred_final['Prospect ID'] = y_train.index

# Displaying head of created dataframe:-

y_train_pred_final.head()


# ### Finding Optimal Cutoff Point

# In[138]:


# Let's create columns with different probability cutoffs :-
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[139]:


#Importing 'metrics' library:-

from sklearn import metrics

# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

from sklearn.metrics import confusion_matrix

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[140]:


#Setting plot style:-

sns.set(style = 'darkgrid')


# In[141]:


# Plotting accuracy, sensitivity and specificity for various probabilities:-


#plt.figure(figsize=(20,5))
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.xticks(np.arange(0,1,step=0.05),size=8)
plt.axvline(x=0.358, color='r', linestyle='--') # additing axline

plt.show()


# ### Note - From the curve above, it seems that 0.358 is optimal cutoff point to take .

# In[142]:


# Calculating 'final_predicted' based on 'Converted_Prob' using 0.358 cutoff point:-

y_train_pred_final['final_predicted'] = y_train_pred_final.Converted_Prob.map( lambda x: 1 if x > 0.358 else 0)


# In[143]:


# Dropping the unnecessary columns:-

y_train_pred_final.drop([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],axis = 1, inplace = True) 


# In[144]:


#Displaying the head:-

y_train_pred_final.head() 


# In[145]:


# Assigning the 'Lead_Score' based on 'Converted_Prob' :-

y_train_pred_final['Lead_Score'] = y_train_pred_final.Converted_Prob.map( lambda x: round(x*100))


# In[146]:


# Selecting only important columns and displaying head of dataframe:-

y_train_pred_final[['Converted','Converted_Prob','Prospect ID','final_predicted','Lead_Score']].head()


# ### Model Evaluation

# In[147]:


#Importing 'metrics' library:-

from sklearn import metrics

# Confusion matrix:-

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted)
print(confusion)


# In[149]:


# Plotting confusion matrix:-
sns.heatmap(confusion, annot=True,fmt='g',cmap='GnBu')
plt.xlabel('Predicted',fontsize=20)
plt.ylabel('Actual',fontsize=20)
plt.show()


# In[150]:


# Check the overall accuracy:-

metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)


# In[151]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[152]:


# Checking the sensitivity of our logistic regression model:-

TP / float(TP+FN)


# In[153]:


# Calculating specificity:-

TN / float(TN+FP)


# In[154]:


# Calculate False Postive Rate - predicting conversion when customer does not have convert:-

print(FP/ float(TN+FP))


# In[155]:


# positive predictive value :-

print (TP / float(TP+FP))


# In[156]:


# Negative predictive value:-

print (TN / float(TN+ FN))


# #### Precision and Recall

# In[157]:


# Precision
confusion[1,1]/(confusion[0,1]+confusion[1,1])


# In[158]:


# Recall
confusion[1,1]/(confusion[1,0]+confusion[1,1])


# ### Note
# 
# We have the following values for the Train Data:
# 
# - Accuracy :    81%
# - Sensitivity : 80%
# - Specificity : 81%
# - Pricision:    73%
# - Recall:       80%

# ### Plotting the ROC Curve

# In[167]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[168]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Converted_Prob, drop_intermediate = False )


# In[163]:


draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_Prob)


# Note - getting a good value of 0.89 indicating a good predictive model.As ROC Curve should be a value close to 1.

# ### Precision and recall tradeoff

# In[171]:


# Importing required library for 'precision_recall_curve' :-

from sklearn.metrics import precision_recall_curve


# In[172]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_Prob)


# In[173]:


plt.figure(figsize=(20,5))
plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.title('Precision Recall Curve',fontsize=20)
plt.axvline(x=0.427, color='b', linestyle='--') # additing axline
plt.xticks(np.arange(0,1,step=0.02),size=10)
plt.yticks(size=20)

plt.show()


# Note - above 'precision_recall_curve' we can see that cutoff point is 0.427

# In[174]:


# plotting the Train dataset again with 0.427 as cutoff:-

y_train_pred_final['final_predicted_2'] = y_train_pred_final['Converted_Prob'].map( lambda x: 1 if x > 0.427 else 0)
y_train_pred_final.head() # Displaying head 


# In[175]:


# Confusion matrix:-

confusion_2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted_2)
print(confusion_2)


# In[176]:


# Plotting confusion matrix:-
sns.heatmap(confusion_2, annot=True,fmt='g',cmap='GnBu')
plt.xlabel('Predicted',fontsize=20)
plt.ylabel('Actual',fontsize=20)
plt.show()


# In[177]:


# Check the overall accuracy:-

metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted_2)


# In[178]:


TP = confusion_2[1,1] # true positive 
TN = confusion_2[0,0] # true negatives
FP = confusion_2[0,1] # false positives
FN = confusion_2[1,0] # false negatives


# In[179]:


# Checking the sensitivity of our logistic regression model:-

TP / float(TP+FN)


# In[180]:


# Calculating specificity:-

TN / float(TN+FP)


# In[181]:


# positive predictive value :-

print (TP / float(TP+FP))


# In[182]:


# Negative predictive value:-

print (TN / float(TN+ FN))


# In[183]:


# Pricision:-

confusion_2[1,1]/(confusion_2[0,1]+confusion_2[1,1])


# In[184]:


# Calculating 'Recall' :-

confusion_2[1,1]/(confusion_2[1,0]+confusion_2[1,1])


# Note - By using the Precision - Recall trade off curve cut off point True Positive number has decrease and True Negative number has increase
# So, cannot be used this precision method

# ### Making predictions on the test set

# In[185]:


# Applying 'transform' to scale the 'test' data set:-

X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.transform(X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])


# In[186]:


# Predicting using values used by the final model i.e. logm3:-

test_col = X_train_sm3.columns

X_test=X_test[test_col[1:]]
# Adding constant variable to test dataframe:-
X_test = sm.add_constant(X_test)

X_test.info() #Displaying info about columns


# In[187]:


# Predicting on test data set using final model :-

y_test_pred = logm3.predict(X_test)


# In[188]:


# Checking top 10 rows:-

y_test_pred[:10]


# In[189]:


# Converting y_test_pred to a dataframe :-

y_pred_1 = pd.DataFrame(y_test_pred)


# In[190]:


# Let's see the head
y_pred_1.head()


# In[191]:


# Converting y_test to dataframe:-

y_test_df = pd.DataFrame(y_test)


# In[192]:


# Putting 'Prospect ID' to index:-

y_test_df['Prospect ID'] = y_test_df.index


# In[193]:


# Removing index for both dataframes to append them side by side :-

y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[194]:


# Appending y_test_df and y_pred_1:-

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[195]:


# Displaying head of 'y_pred_final' :-

y_pred_final.head()


# In[196]:


# Renaming the column '0' as 'Converted_Prob':-

y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_Prob'})


# In[197]:


# Rearranging the columns:-

y_pred_final = y_pred_final[['Prospect ID','Converted','Converted_Prob']]


# In[198]:


# Let's see the head of y_pred_final:-

y_pred_final.head()


# In[199]:


#Assigning 'Lead Score' to dataframe 'y_pred_final':-

y_pred_final['Lead_Score'] = y_pred_final.Converted_Prob.map( lambda x: round(x*100))


# In[200]:


y_pred_final.head()


# In[201]:


# Calculating 'final_Predicted' based on 'Converted_Prob' for cutoff point 0.357:-

y_pred_final['final_Predicted'] = y_pred_final.Converted_Prob.map(lambda x: 1 if x > 0.357 else 0)


# In[202]:


# Displaying the head of 'y_pred_final' dataframe:-

y_pred_final.head()


# In[203]:


# Checking the overall accuracy:-

metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_Predicted)


# In[204]:


# Calculating confusion matrix for test data:-

confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_Predicted )
confusion2


# In[205]:


# Plotting confusion matrix:-
sns.heatmap(confusion2, annot=True,fmt='g',cmap='GnBu')
plt.xlabel('Predicted',fontsize=20)
plt.ylabel('Actual',fontsize=20)
plt.show()


# In[206]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[207]:


# Let's see the sensitivity of our logistic regression model:-

TP / float(TP+FN)


# In[208]:


# Let us calculate specificity:-

TN / float(TN+FP)


# In[209]:


# Calculate False Postive Rate - predicting conversion when customer does not have convert:-

print(FP/ float(TN+FP))


# In[210]:


# positive predictive value :-

print (TP / float(TP+FP))


# In[211]:


# Negative predictive value:-

print (TN / float(TN+ FN))


# In[213]:


#Importing 'precision_score' and 'recall_score':_

from sklearn.metrics import precision_score, recall_score


# In[214]:


# Calculating 'precision_score':-

precision_score(y_pred_final.Converted, y_pred_final.final_Predicted)


# In[215]:


#Calculating 'recall_score':-

recall_score(y_pred_final.Converted, y_pred_final.final_Predicted)


# Note:
#     
# - The sensitivity value for test data is 80% while for train data is also 80% . The accuracy values is ~ 80%. Which shows that model is performing well for test data set also.

# ### Determining top feature based on final model (logm3)

# In[216]:


# Displaying parameters obtained by final model 'logm3':-

pd.options.display.float_format = '{:.2f}'.format # Setting format option
logm3.params[1:]


# In[217]:


#Getting a relative coeffient value for all the features wrt the feature with the highest coefficient:-

top_feature = logm3.params[1:]
top_feature = 100.0 * (top_feature / top_feature.max())
top_feature


# In[218]:


# Plotting the feature variables based on their relative importance:-

top_feature_sort = np.argsort(top_feature,kind='quicksort',order='list of str')

pos = np.arange(top_feature_sort.shape[0]) + .5

fig1 = plt.figure(figsize=(10,5))
ax = fig1.add_subplot(1, 1, 1)
ax.barh(pos, top_feature[top_feature_sort])
ax.set_yticks(pos)
ax.set_yticklabels(np.array(X_train_new.columns)[top_feature_sort], fontsize=13)
ax.set_xlabel('Top Features', fontsize=15)
plt.show()


# ## Recomendation:
# 
# - Lead Source_Welingak Website : As conversion rate is higher for those leads who got to know about course from 'Welingak Website',so company can focus on this website to get more number of potential leads.
# 
# - Lead Origin_Lead Add Form: Leads who have engaged through 'Lead Add Form' having higher conversion rate so company can focus on it to get more number of leads cause have a higher chances of getting converted.
# 
# - Last Activity_SMS Sent: Lead whose last activity is sms sent can be potential lead for company.
# 
# - Total Time Spent on website: Leads spending more time on website can be the potential lead.

# In[ ]:




