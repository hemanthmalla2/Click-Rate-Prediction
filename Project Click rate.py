#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
from xgboost import XGBRegressor
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


pd.set_option('display.max_columns',None)


# In[3]:


df=pd.read_csv("train.csv")
df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df_test=pd.read_csv('test.csv')
df_test.head()


# # Looking for missing values

# In[7]:


missing = pd.DataFrame()
missing["sum"]=df.isna().sum()
missing["percentage"]= df.isna().mean() * 100
missing=missing[missing["sum"]>0].sort_values(ascending=False,by='sum')
missing


# * so no missing values
# * looking into head i can say campain_id is not so useful so drop it

# In[8]:


df.drop("campaign_id",axis=1,inplace=True)


# In[9]:


df.head()


# * we will split data into `x,y` and `x` further into `continuios, catagorical and boolean` data 

# In[10]:


x=df.drop("click_rate",axis=1)
y=df.click_rate


# In[11]:


unique=[]
bool_col=[]
cat_col=[]
cont_col =[]


# In[12]:


for i in x.columns:
    n_values = x[i].nunique()
    if n_values==1:
        unique.append(i)
    elif n_values == 2:
        bool_col.append(i)
    elif n_values < 10:
        cat_col.append(i)
    else:
        cont_col.append(i)


# In[13]:


print("Unique columns in the data are :",unique,end='\n \n')
print("Boolean columns in the data are :",bool_col,end='\n \n')
print("Catagorical columns in the data are :",cat_col,end='\n \n')
print("Continuious columns in the data are :",cont_col,end='\n \n')


# In[14]:


x.drop(unique,axis=1,inplace=True)


# In[15]:


df_test.drop(unique,axis=1,inplace=True)


# In[16]:


x.head()


# In[17]:


x.shape


# # boolean Data

# In[18]:


x[bool_col].head()


# # Catagorical Columns

# In[19]:


x[cat_col].head()


# * `day_of_week,is_image, is_quote, is_emotiocons` looks numerial but still it is undr catagorical section
# * `is_price` looks boolean but are in catagorical section 
# * lets analyze the data'

# In[20]:


sns.countplot(x=x.day_of_week)


# In[21]:


sns.boxplot(x=x.day_of_week,y=y)


# In[ ]:





# * Though it looks numerical data it is actually catagorical by nature 
# * so we will convert tthis to catagorical data

# In[22]:


x.day_of_week=x.day_of_week.astype('object')


# In[23]:


df_test.day_of_week=df_test.day_of_week.astype('object')


# In[24]:


# is_image
sns.countplot(x.is_image)


# * is_image if no of image >2 is very minor in share so we will group them as a feature

# In[25]:


x.is_image= x.is_image.apply(lambda x: 3 if x>2 else x)
df_test.is_image=df_test.is_image.apply(lambda x: 3 if x>2 else x)


# In[26]:


sns.countplot(x.is_image)


# In[27]:


# is_quote
sns.countplot(x.is_quote)


# * if is_quote> 3 we will group them as feature we will impute 100 as other option to have an understanding

# In[28]:


x.is_quote =x.is_quote.apply(lambda x: 100 if x>3 else x)
df_test.is_quote=df_test.is_quote.apply(lambda x: 100 if x>3 else x)


# In[29]:


sns.countplot(x.is_quote)


# In[30]:


# is_emoticons
sns.countplot(x.is_emoticons)


# * data is very skewed we will make  is if 0 else 1

# In[31]:


x.is_emoticons = x.is_emoticons.apply(lambda x : 0 if x==0 else 1)
df_test.is_emoticons=df_test.is_emoticons.apply(lambda x : 0 if x==0 else 1)


# In[32]:


sns.countplot(x.is_emoticons)


# * converting day_of_week,is_image, is_quote, is_emotiocons to object type

# In[33]:


# day_of_week, is_image, is_quote, is_emotiocons
x[["is_image", "is_quote", "is_emoticons"]] =x[["is_image", "is_quote", "is_emoticons"]].astype('object')
df_test[["is_image", "is_quote", "is_emoticons"]]=df_test[["is_image", "is_quote", "is_emoticons"]].astype('object')


# In[34]:


# is_image
sns.countplot(x.is_price)


# In[35]:


# is_image
sns.countplot(x.is_price)


# In[36]:


x.is_price.value_counts(normalize=True)


# * looking into the data the is totally useless as the data is highly skewed so we will drop it 

# In[37]:


x.drop('is_price',axis=1,inplace=True)
df_test.drop('is_price',axis=1,inplace=True)
cat_col.remove('is_price')


# * we will One hot encode other varibales later we will perform rfe to select kbest out of them

# In[38]:


x[cat_col].head()


# In[39]:


for i in cat_col:
    dummy=pd.get_dummies(x[i],prefix=i)
    x= pd.concat([x,dummy],axis=1)
    test_dummy=pd.get_dummies(df_test[i],prefix=i)
    df_test = pd.concat([df_test,test_dummy],axis=1)
    x.drop(i,axis=1,inplace=True)
    df_test.drop(i,axis=1,inplace=True)            


# In[40]:


x.head()


# In[41]:


x.shape


# # continuious data

# In[42]:


x[cont_col].head()


# In[43]:


sns.countplot(x.sender)


# * though the data looks continuious it acts catagorical and is skewed
# * we will group 2, 10, 15 and all other as 1 

# In[44]:


x.sender= x.sender.apply(lambda x: 100 if x not in [3,10,15] else x )


# In[45]:


df_test.sender=df_test.sender.apply(lambda x: 100 if x not in [3,10,15] else x )


# In[46]:


sns.countplot(x.sender)


# In[47]:


x.sender=x.sender.astype('object')
df_test.sender=df_test.sender.astype('object')


# In[48]:


dummy=pd.get_dummies(x.sender,prefix='sender')
x=pd.concat([x,dummy],axis=1)
x.drop('sender',axis=1,inplace=True)


# In[49]:


dummy=pd.get_dummies(df_test.sender,prefix='sender')
df_test=pd.concat([df_test,dummy],axis=1)
df_test.drop('sender',axis=1,inplace=True)


# In[50]:


fig,axs= plt.subplots(ncols=3,figsize=(35,5))
sns.kdeplot(x.subject_len,ax=axs[0])
sns.violinplot(x.subject_len,ax=axs[1])
sns.boxplot(x.subject_len,ax=axs[2])


# In[51]:


# body_len
fig,axs= plt.subplots(ncols=3,figsize=(35,5))
sns.kdeplot(x.body_len,ax=axs[0])
sns.violinplot(x.body_len,ax=axs[1])
sns.boxplot(x.body_len,ax=axs[2])


# In[52]:


# mean_paragraph_len
fig,axs= plt.subplots(ncols=3,figsize=(35,5))
sns.kdeplot(x.mean_paragraph_len,ax=axs[0])
sns.violinplot(x.mean_paragraph_len,ax=axs[1])
sns.boxplot(x.mean_paragraph_len,ax=axs[2])


# In[53]:


# Category
sns.countplot(x.category)


# * we will goorup catagories with 1,2,6,9,10,15 and others

# In[54]:


x.category = x.category.apply(lambda x: 100 if x not in [1,2,6,9,10,15] else x)
df_test.category=df_test.category.apply(lambda x: 100 if x not in [1,2,6,9,10,15] else x)


# In[55]:


sns.countplot(x.category)


# In[56]:


x.category=x.category.astype('object')
df_test.category=df_test.category.astype('object')


# In[57]:


dummy=pd.get_dummies(x.category,prefix='category')
x=pd.concat([x,dummy],axis=1)
x.drop('category',axis=1,inplace=True)


# In[58]:


dummy=pd.get_dummies(df_test.category,prefix='category')
df_test=pd.concat([df_test,dummy],axis=1)
df_test.drop('category',axis=1,inplace=True)


# In[59]:


# mean_paragraph_len
fig,axs= plt.subplots(ncols=3,figsize=(35,5))
sns.kdeplot(x['product'],ax=axs[0])
sns.violinplot(x['product'],ax=axs[1])
sns.boxplot(x['product'],ax=axs[2])


# * Looking at the kde plot and violen plot looks like catagorical data by nature

# In[60]:


sns.countplot(x['product'])


# * This is a catagorical varibale but it has 50 features
# * we will convert them to binary but later point we will use RFE to reduce feture selection

# In[61]:


prod=x['product'].value_counts().reset_index()
prod=prod[prod["product"] > 50]
prod=list(prod['index'])
prod


# * now we use prod to transform the data

# In[62]:


x['product']=x['product'].apply(lambda x: 100 if x not in prod else x)
df_test['product']=df_test['product'].apply(lambda x: 100 if x not in prod else x)


# In[63]:


df_test['product']=df_test['product'].astype('object')
df_test['product']=df_test['product'].astype('object')


# In[64]:


sns.countplot(x['product'])


# In[65]:


dummy=pd.get_dummies(x['product'],prefix='product')
x=pd.concat([x,dummy],axis=1)
x.drop('product',axis=1,inplace=True)


# In[66]:


dummy=pd.get_dummies(df_test['product'],prefix='product')
df_test=pd.concat([df_test,dummy],axis=1)
df_test.drop('product',axis=1,inplace=True)


# In[67]:


# no_of_CTA
fig,axs= plt.subplots(ncols=3,figsize=(35,5))
sns.kdeplot(x.no_of_CTA,ax=axs[0])
sns.violinplot(x.no_of_CTA,ax=axs[1])
sns.boxplot(x.no_of_CTA,ax=axs[2])


# In[68]:


# mean_CTA_len
fig,axs= plt.subplots(ncols=3,figsize=(35,5))
sns.kdeplot(x.mean_CTA_len,ax=axs[0])
sns.violinplot(x.mean_CTA_len,ax=axs[1])
sns.boxplot(x.mean_CTA_len,ax=axs[2])


# In[69]:


# target_audience
fig,axs= plt.subplots(ncols=3,figsize=(35,5))
sns.kdeplot(x.target_audience,ax=axs[0])
sns.violinplot(x.target_audience,ax=axs[1])
sns.boxplot(x.target_audience,ax=axs[2])


# In[70]:


sns.countplot(x.target_audience)


# In[71]:


x.target_audience= x.target_audience.apply(lambda x : 100 if x not in [10,12,14,15,16] else x)
df_test.target_audience= df_test.target_audience.apply(lambda x : 100 if x not in [10,12,14,15,16] else x)


# In[72]:


x.target_audience = x.target_audience.astype('object')
df_test.target_audience = df_test.target_audience.astype('object')


# In[73]:


sns.countplot(x.target_audience)


# In[74]:


dummy=pd.get_dummies(x['target_audience'],prefix='target_audience')
x=pd.concat([x,dummy],axis=1)
x.drop('target_audience',axis=1,inplace=True)


# In[75]:


dummy=pd.get_dummies(df_test['target_audience'],prefix='target_audience')
df_test=pd.concat([df_test,dummy],axis=1)
df_test.drop('target_audience',axis=1,inplace=True)


# In[76]:


x.head()


# In[77]:


x.shape


# In[78]:


plt.figure(figsize=(25,15))
sns.heatmap(x.corr(),vmin= -1,vmax=1)


# In[ ]:





# # Remove multi colinearty within the data

# In[79]:


num_col=[]
for i in x.columns:
    if x[i].nunique() !=2:
        num_col.append(i)


# In[80]:


num_col


# In[81]:


print("numerical columns after mulit colinearity removal are :",num_col)


# In[82]:


print("we have", len(num_col),"numerical columns")


# * we need to scale them

# In[83]:


x[num_col].head()


# In[84]:


corr=pd.concat([x[num_col],y],axis=1)
corr=corr.corr()
sns.heatmap(corr,annot=True,vmax=1,vmin = -1)


# In[85]:


corr=pd.concat([x[num_col],np.log(y)],axis=1)
corr=corr.corr()
sns.heatmap(corr,annot=True,vmax=1,vmin = -1)


# * there is no high coo-relation with meand Linear regression doesnt perform wel even after tranformting y column to normal distribution
# * so we will build xgboost regression model
# * XgBoost is capable of handling Mulit colinearty

# In[86]:


x[num_col].head()


# In[87]:


scale= StandardScaler()
x[num_col]=scale.fit_transform(x[num_col])
df_test[num_col]=scale.transform(df_test[num_col])


# In[88]:


x[num_col].head()


# In[89]:


df_test[num_col].head()


# In[90]:


# PCA can handle multi colinearity so we will transform the entire data with multicolinearity
pca=PCA()


# In[91]:


pca.fit(x)


# In[92]:


len(pca.explained_variance_ratio_)


# In[93]:


np.round(pca.explained_variance_ratio_ ,4)


# In[94]:


sns.lineplot(x=[i for i in range(1,60)],y=pca.explained_variance_ratio_)


# In[95]:


# PCA can handle multi colinearity so we will transform the entire data with multicolinearity
pca=PCA(0.95)


# In[96]:


pca.fit(x)


# In[97]:


len(pca.explained_variance_ratio_)


# In[98]:


np.round(pca.explained_variance_ratio_ ,4)


# In[99]:


pac_data= pca.transform(x)


# In[100]:


Y= np.log(y)
Y


# In[101]:


get_ipython().run_line_magic('pinfo', 'XGBRegressor')


# In[102]:


def objective(trial):
    train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.2,random_state=42)
    
    # To select which parameters to optimize, please look at the XGBoost documentation:
    # https://xgboost.readthedocs.io/en/latest/parameter.html
    param = {
        'lambda': trial.suggest_loguniform(
            'lambda', 1e-3, 10.0
        ),
        'alpha': trial.suggest_loguniform(
            'alpha', 1e-3, 10.0
        ),
        'colsample_bytree': trial.suggest_categorical(
            'colsample_bytree', [0.5,0.6,0.7,0.8,0.9,1.0]
        ),
        'subsample': trial.suggest_categorical(
            'subsample', [0.6,0.7,0.8,1.0]
        ),
        'learning_rate': trial.suggest_float(
            'learning_rate', 0.001,0.4
        ),
        'n_estimators': trial.suggest_int(
            "n_estimators", 150,4000
        ),
        'max_depth': trial.suggest_categorical(
            'max_depth', [4,5,7,9,11,13,15,17]
        ),
        'random_state': 42,
        'min_child_weight': trial.suggest_int(
            'min_child_weight', 1, 300
        ),
    }
    model = XGBRegressor(**param)  
    model.set_params(early_stopping_rounds=100)
    model.fit(train_x,train_y,eval_set=[(test_x,test_y)],verbose=False)
    
    preds = model.predict(test_x)
    
    mae = mean_absolute_error(test_y, preds)
    #print(r2)
    return mae


# In[103]:


study = optuna.create_study(direction='minimize')
study.optimize(objective,show_progress_bar=True,n_trials=250)
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)


# In[ ]:


# 0.5633683859736183
#params={'lambda': 0.015311581534899784, 'alpha': 0.001184792602113157, 'colsample_bytree': 1.0, 'subsample': 1.0, 'learning_rate': 0.16734864054814122, 'n_estimators': 2000, 'max_depth': 19, 'min_child_weight': 53}


# In[104]:


# rms  0.04546945700255865.
#params={'lambda': 2.516667908895054, 'alpha': 0.005662479324960345, 'colsample_bytree': 0.9, 'subsample': 1.0, 'learning_rate': 0.2466907779941183, 'n_estimators': 200, 'max_depth': 20, 'min_child_weight': 59}


# In[ ]:


#mse
#params= {'lambda': 2.5482467578467127, 'alpha': 0.010543549331554023, 'colsample_bytree': 0.9, 'subsample': 1.0, 'learning_rate': 0.02423586659801208, 'n_estimators': 3000, 'max_depth': 14, 'min_child_weight': 1}


# In[107]:


# mse
#params={'lambda': 8.245290407832472, 'alpha': 0.02005842343151029, 'colsample_bytree': 0.9, 'subsample': 0.6, 'learning_rate': 0.18284853465775586, 'n_estimators': 3000, 'max_depth': 13, 'min_child_weight': 1}


# In[104]:


# mse 0.026446392019721706.
params={'lambda': 9.993483170541582, 'alpha': 0.02234754741590284, 'colsample_bytree': 0.7, 'subsample': 0.8, 'learning_rate': 0.00614994012757663, 'n_estimators': 3883, 'max_depth': 15, 'min_child_weight': 1}


# In[105]:


model= XGBRegressor(**params)


# In[106]:


model.fit(x,y)


# In[107]:


y_pred=model.predict(df_test.drop('campaign_id',axis=1))


# In[108]:


y_pred=y_pred.round(6)
y_pred


# In[109]:


x_pred=model.predict(x)


# In[110]:


x_pred


# In[111]:


r2_score(y,x_pred)


# In[112]:


submission= pd.DataFrame()


# In[113]:


submission["campaign_id"]=df_test.campaign_id


# In[114]:


submission["click_rate"] = y_pred 


# In[115]:


submission[submission['click_rate']<0]


# In[116]:


submission.shape


# In[117]:


submission['click_rate']=submission['click_rate'].apply(lambda x: abs(x) if x<0 else x)


# In[118]:


submission[submission['click_rate']<0]['click_rate']


# In[119]:


submission.shape


# In[120]:


submission.to_csv("Submission-5.csv",index=False)


# In[121]:


submission[submission['campaign_id'] == 2021]


# In[ ]:




