import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

#read train data and create dataframe for train data
train_df = pd.read_csv('../train.csv')

#NA is not 'Not available'.It is 'No Alley access'.So fill that values with 'No Alley'
train_df['Alley'].fillna(value='No Alley',inplace=True)

train_df['Alley'].value_counts()

columns = train_df.columns
#This is changed
#find correlation for all the columns and work on high values
corr = train_df.corr()

#change integer categorical values to categorical values by mapping the integers to string.
train_df['OverallQual'] = train_df.OverallQual.astype('category')
train_df['OverallQual'] = train_df.OverallQual.map({10:'Very Excellent',9:'Excellent',8:'Very Good',7:'Good',6:'Above Average',5:'Average',4:'Beow Average',3:'Fair',2:'Poor',1:'Very Poor'})

train_df['OverallCond'] = train_df.OverallCond.astype('category')
train_df['OverallCond'] = train_df.OverallCond.map({10:'Very Excellent',9:'Excellent',8:'Very Good',7:'Good',6:'Above Average',5:'Average',4:'Beow Average',3:'Fair',2:'Poor',1:'Very Poor'})

train_df['MSSubClass'] = train_df.MSSubClass.astype('category')
train_df['MSSubClass'] = train_df.MSSubClass.map({20:'1-STORY 1946 & NEWER ALL STYLES',30:'1-STORY 1945 & OLDER',40:'1-STORY W/FINISHED ATTIC ALL AGES',45:'1-1/2 STORY - UNFINISHED ALL AGES',50:'1-1/2 STORY FINISHED ALL AGES',60:'2-STORY 1946 & NEWER',70:'2-STORY 1945 & OLDER',75:'2-1/2 STORY ALL AGES',80:'SPLIT OR MULTI-LEVEL',85:'SPLIT FOYER',90:'DUPLEX - ALL STYLES AND AGES',120:'1-STORY PUD (Planned Unit Development) - 1946 & NEWER',150:'1-1/2 STORY PUD - ALL AGES',160:'2-STORY PUD - 1946 & NEWER',180:'PUD - MULTILEVEL - INCL SPLIT LEV/FOYER',190:'2 FAMILY CONVERSION - ALL STYLES AND AGES'})

#NA is not 'Not available'.It is 'No Basement'.So fill that values with 'No Basement'
train_df['BsmtQual'].fillna(value='No Basement',inplace=True)
train_df['BsmtQual'].value_counts()

train_df['BsmtCond'].fillna(value='No Basement',inplace=True)
train_df['BsmtCond'].value_counts()

train_df['BsmtExposure'].fillna(value='No Basement',inplace=True)
train_df['BsmtExposure'].value_counts()

train_df['BsmtFinType1'].fillna(value='No Basement',inplace=True)
train_df['BsmtFinType1'].value_counts()

train_df['BsmtFinType2'].fillna(value='No Basement',inplace=True)
train_df['BsmtFinType2'].value_counts()

#NA is not 'Not available'.It is 'No Fire place'.So fill that values with 'No Fire place'
train_df['FireplaceQu'].fillna(value='No Fireplace',inplace=True)
train_df['FireplaceQu'].value_counts()

#NA is not 'Not available'.It is 'No Garage'.So fill that values with 'No Garage'
train_df['GarageType'].fillna(value='No Garage',inplace=True)
train_df['GarageType'].value_counts()

train_df['GarageFinish'].fillna(value='No Garage',inplace=True)
train_df['GarageFinish'].value_counts()

train_df['GarageQual'].fillna(value='No Garage',inplace=True)
train_df['GarageQual'].value_counts()

train_df['GarageCond'].fillna(value='No Garage',inplace=True)
train_df['GarageCond'].value_counts()

#NA is not 'Not available'.It is 'No Pool'.So fill that values with 'No Pool'
train_df['PoolQC'].fillna(value='No Pool',inplace=True)
train_df['PoolQC'].value_counts()

#NA is not 'Not available'.It is 'No Fence'.So fill that values with 'No Fence'
train_df['Fence'].fillna(value='No Fence',inplace=True)
train_df['Fence'].value_counts()

train_df['MiscFeature'].fillna(value='None',inplace=True)
train_df['MiscFeature'].value_counts()

#Categorize the categorical and non-categorical data.Earlier we converted 'OverallQual' and 'OverallCond' to dtype(category).To easily categorize, convert them again to dtype(object)

train_df['OverallQual'] = train_df.OverallQual.astype('object')
train_df['OverallCond'] = train_df.OverallQual.astype('object')
train_df['MSSubClass'] = train_df.MSSubClass.astype('object')


cat = []
non_cat = []
for column in columns:
    if(train_df[column].dtype == 'O'):
        cat.append(column)
    else:
        non_cat.append(column)
        
#check for missing values in cat data.
cat_na = train_df[cat].isna().sum()   

#check for the maximum values is MasVnrType.It is 'None'.Replace missing values with 'None'
train_df['MasVnrType'].mode()   
train_df['MasVnrType'].fillna(value='None',inplace=True)
train_df['MasVnrType'].isna().sum()

#Do the same for Electrical.
train_df['Electrical'].mode()   
train_df['Electrical'].fillna(value='SBrkr',inplace=True)
train_df['Electrical'].isna().sum()

#check for missing values in non_cat data.
non_cat_na = train_df[non_cat].isna().sum()   

#Most of the vaues of GarageYrBlt are same as YearBuilt,so fillna with those.
train_df['GarageYrBlt'].fillna(value=train_df['YearBuilt'],inplace=True)

train_df['LotFrontage'].fillna(value=train_df['LotFrontage'].mean(),inplace=True)

#create dummies to convert categorical data to numerical data
dummy_df = pd.get_dummies(train_df[cat],drop_first=True,prefix=cat)

dummy_and_noncat_df = pd.concat([train_df[non_cat],dummy_df],axis=1)

for column in dummy_and_noncat_df.columns:
    if (dummy_and_noncat_df[column].corr(dummy_and_noncat_df['SalePrice']))<0.01:
        dummy_and_noncat_df.drop(column)

#remove SalePrice to make it train data.
non_cat.remove('SalePrice')

#make a final df that includes categorical dummies and non cat data.
train_df_final = pd.concat([train_df[non_cat],dummy_df],axis=1)

train_df_final

X_train  = train_df_final
y_train =  train_df['SalePrice']



#.......................WORKING ON TEST DATA...................................
test_df = pd.read_csv('../test.csv')

test_df['Alley'].fillna(value='No Alley',inplace=True)

test_df['Alley'].value_counts()

columns_test = test_df.columns

#change integer categorical values to categorical values by mapping the integers to string.
test_df['OverallQual'] = train_df.OverallQual.astype('category')
test_df['OverallQual'] = test_df.OverallQual.map({10:'Very Excellent',9:'Excellent',8:'Very Good',7:'Good',6:'Above Average',5:'Average',4:'Beow Average',3:'Fair',2:'Poor',1:'Very Poor'})

test_df['OverallCond'] = test_df.OverallCond.astype('category')
test_df['OverallCond'] = test_df.OverallCond.map({10:'Very Excellent',9:'Excellent',8:'Very Good',7:'Good',6:'Above Average',5:'Average',4:'Beow Average',3:'Fair',2:'Poor',1:'Very Poor'})

test_df['MSSubClass'] = test_df.MSSubClass.astype('category')
test_df['MSSubClass'] = test_df.MSSubClass.map({20:'1-STORY 1946 & NEWER ALL STYLES',30:'1-STORY 1945 & OLDER',40:'1-STORY W/FINISHED ATTIC ALL AGES',45:'1-1/2 STORY - UNFINISHED ALL AGES',50:'1-1/2 STORY FINISHED ALL AGES',60:'2-STORY 1946 & NEWER',70:'2-STORY 1945 & OLDER',75:'2-1/2 STORY ALL AGES',80:'SPLIT OR MULTI-LEVEL',85:'SPLIT FOYER',90:'DUPLEX - ALL STYLES AND AGES',120:'1-STORY PUD (Planned Unit Development) - 1946 & NEWER',150:'1-1/2 STORY PUD - ALL AGES',160:'2-STORY PUD - 1946 & NEWER',180:'PUD - MULTILEVEL - INCL SPLIT LEV/FOYER',190:'2 FAMILY CONVERSION - ALL STYLES AND AGES'})

#NA is not 'Not available'.It is 'No Basement'.So fill that values with 'No Basement'
test_df['BsmtQual'].fillna(value='No Basement',inplace=True)
test_df['BsmtQual'].value_counts()

test_df['BsmtCond'].fillna(value='No Basement',inplace=True)
test_df['BsmtCond'].value_counts()

test_df['BsmtExposure'].fillna(value='No Basement',inplace=True)
test_df['BsmtExposure'].value_counts()

test_df['BsmtFinType1'].fillna(value='No Basement',inplace=True)
test_df['BsmtFinType1'].value_counts()

test_df['BsmtFinType2'].fillna(value='No Basement',inplace=True)
test_df['BsmtFinType2'].value_counts()

#NA is not 'Not available'.It is 'No Fire place'.So fill that values with 'No Fire place'
test_df['FireplaceQu'].fillna(value='No Fireplace',inplace=True)
test_df['FireplaceQu'].value_counts()

#NA is not 'Not available'.It is 'No Garage'.So fill that values with 'No Garage'
test_df['GarageType'].fillna(value='No Garage',inplace=True)
test_df['GarageType'].value_counts()

test_df['GarageFinish'].fillna(value='No Garage',inplace=True)
test_df['GarageFinish'].value_counts()

test_df['GarageQual'].fillna(value='No Garage',inplace=True)
test_df['GarageQual'].value_counts()

test_df['GarageCond'].fillna(value='No Garage',inplace=True)
test_df['GarageCond'].value_counts()

#NA is not 'Not available'.It is 'No Pool'.So fill that values with 'No Pool'
test_df['PoolQC'].fillna(value='No Pool',inplace=True)
test_df['PoolQC'].value_counts()

#NA is not 'Not available'.It is 'No Fence'.So fill that values with 'No Fence'
test_df['Fence'].fillna(value='No Fence',inplace=True)
test_df['Fence'].value_counts()

test_df['MiscFeature'].fillna(value='None',inplace=True)
test_df['MiscFeature'].value_counts()

#Categorize the categorical and non-categorical data.Earlier we converted 'OverallQual' and 'OverallCond' to dtype(category).To easily categorize, convert them again to dtype(object)

test_df['OverallQual'] = train_df.OverallQual.astype('object')
test_df['OverallCond'] = train_df.OverallQual.astype('object')
test_df['MSSubClass'] = train_df.MSSubClass.astype('object')


test_cat = []
test_non_cat = []
for column in columns_test:
    if(test_df[column].dtype == 'O'):
        test_cat.append(column)
    else:
        test_non_cat.append(column)
        
#check for missing values in cat data.
test_cat_na = test_df[test_cat].isna().sum()   

for column in test_cat:
    if(test_df[column].isna().sum() != 0):
        test_df[column].fillna(train_df[column].mode()[0],inplace=True)
        
#check for missing values in non_cat data.
non_cat_na = test_df[non_cat].isna().sum()   

#Most of the vaues of GarageYrBlt are same as YearBuilt,so fillna with those.
test_df['GarageYrBlt'].fillna(value=train_df['YearBuilt'],inplace=True)

for column in test_non_cat:
    if(test_df[column].isna().sum() != 0):
        test_df[column].fillna(train_df[column].mean(),inplace=True)

#create dummies to convert categorical data to numerical data
dummy_df_test = pd.get_dummies(test_df[test_cat],drop_first=True,prefix=test_cat)

dummy_df_test

#concat both non_cat and dummy data to make a final df.
test_df_final=pd.concat([test_df[test_non_cat],dummy_df_test],axis=1)

#Use XGBoost 
from xgboost import XGBRegressor

param_dict = {'n_estimators' : [10,20,30], 'max_depth' : [3,4,5], 'min_samples_leaf' : [4,5,6], 'n_jobs' : [10,20,30]}

from sklearn.model_selection import GridSearchCV
GSCV = GridSearchCV(estimator=XGBRegressor(),param_grid=param_dict)

#fit the model in GS to find best estimator.
GSCV.fit(X_train,y_train)
GSCV.best_estimator_

xgbr=XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,colsample_bynode=1, colsample_bytree=1, gamma=0,importance_type='gain', learning_rate=0.1, max_delta_step=0,max_depth=4, min_child_weight=1, min_samples_leaf=4, missing=None,n_estimators=30, n_jobs=10, nthread=None, objective='reg:linear',random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,seed=None, silent=None, subsample=1, verbosity=1)
xgbr.fit(X_train.values,y_train)

#check for no. of columns in train and test, if not equal, make them.
train_df_final.shape
test_df_final.shape

different_columns = set(train_df_final.columns).difference(set(test_df_final.columns))

for column in different_columns:
    test_df_final[column] = 0
    
y_predict = xgbr.predict(test_df_final.values)

df_submit_4 = pd.DataFrame({'Id':test_df['Id'],'SalePrice':y_predict})

df_submit_4.to_csv('MySubmission4',index=False)








        
        











