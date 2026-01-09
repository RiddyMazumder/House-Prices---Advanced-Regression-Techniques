# Generated from: part1house.ipynb
# Converted at: 2026-01-09T20:08:15.580Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # 🏠 House Prices: Advanced Regression Techniques (Part 1)     .......................................................................................................![icons8-spaceship-64.png](attachment:47cc912b-75fc-4ee9-a64a-ce06c4e606cb.png) 
# >  **Predict sales prices and practice feature engineering, RFs, and gradient boosting**                  
# ## Author: RIDDY MAZUMDER
# ## 🔗 Connect with Me
# > [![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/riddymazumder)
# > [![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/RiddyMazumder)
# > [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/riddy-mazumder-7bab46338/)
# > [![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:riddymazumder1971@gmail.com)
# 
# ## Description 
# **This notebook follows a complete end-to-end data science workflow, from loading data to model evaluation and final submission.**  
# ****Each section is clearly explained and well-structured for learning and presentation.****


# ## 1. Libraries Required
# 
# ****In this section, we import all the necessary Python libraries used throughout the project.****  
# **These include libraries for**:
# - **Data manipulation**  
# - **Visualization**
# - **Data Preprocessing**
# - **Machine learning**


import numpy as np # linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# ## 2. Load Dataset


df=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
df.head()

# ## 3. Data Exploration & Cleaning
# 
# ## 3.1 Overview
# 
# **Check shape, missing values, data types.**


df['MSZoning'].value_counts()

df.info()

# ## 3.2 Visualization


sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()
missing_data = pd.DataFrame({
    'Missing_Values': df.isnull().sum(),
    'Percentage': (df.isnull().sum() / len(df)) * 100,
    'Data_Type': df.dtypes
})

missing_data = missing_data[missing_data['Missing_Values'] > 0]

missing_data = missing_data.sort_values(by='Missing_Values', ascending=False)

print("🔍 Missing Data Summary:\n")
display(missing_data)

# ## 3.3 Filling missing values


df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())
df['GarageYrBlt']=df['GarageYrBlt'].fillna(df['GarageYrBlt'].mean())
df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mean())



df.drop(['Alley'],axis=1,inplace=True)
df.drop(['PoolQC','MiscFeature','Fence'],axis=1,inplace=True)

df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])
df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['BsmtFinType1']=df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0])
df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])
df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])
df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['Electrical']=df['Electrical'].fillna(df['Electrical'].mode()[0])
df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])

df.drop(['Id'],axis=1,inplace=True)

df.shape

def check_missing(df):
    # Summary table
    missing_data = pd.DataFrame({
        'Missing_Values': df.isnull().sum(),
        'Percentage': (df.isnull().sum() / len(df)) * 100,
        'Data_Type': df.dtypes
    })
    missing_data = missing_data[missing_data['Missing_Values'] > 0]
    missing_data = missing_data.sort_values(by='Missing_Values', ascending=False)
    
    # Print + display
    print("🔍 Missing Data Summary:\n")
    display(missing_data)
    
    # Heatmap
    plt.figure(figsize=(10,6))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
    plt.title("Missing Values Heatmap")
    plt.show()
check_missing(df)


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df.isnull().sum()

# # 3.4 Remove irrelevant columns


df.dropna(inplace=True)

df.shape

# # 3.5 Encoding 


columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']

len(columns)

def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final

main_df=df.copy

# # Copying data from part 2


test_df=pd.read_csv('/kaggle/input/newdf-test2-csv/test_new.CSV')

test_df.shape

final_df=pd.concat([df,test_df],axis=0)

final_df['SalePrice']

final_df.shape

final_df=category_onehot_multcols(columns)

final_df.shape

final_df =final_df.loc[:,~final_df.columns.duplicated()]

final_df.shape

df_Train=final_df.iloc[:1422,:]
df_Test=final_df.iloc[1422:,:]

df_Train.head()

df_Test.head()

df_Train.shape

# # 3.6 Remove irrelevant columns


df_Test.drop(['SalePrice'],axis=1,inplace=True)

# ## 4. Model Building
# **Libraries Required**


from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# # 4.1 Split Data,Train Model,Evaluate Model


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np

# Features and target
X = df_Train.drop(['SalePrice'], axis=1)
y = df_Train['SalePrice']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize XGBoost regressor
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R² Score:", r2)


# 


# ## 5. Model Accuracy_Score
# **Predictions on training data**


y_pred = classifier.predict(df_Test)

y_pred

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# # 6. Submission File


## Create Sample Submission file and Submit
pred = pd.DataFrame(y_pred)
sub_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
datasets = pd.concat([sub_df['Id'], pred], axis=1)
datasets.columns = ['Id', 'SalePrice']
datasets.to_csv('sample_submission.csv', index=False)


# # Delete Submission File(If anything go wrong)


import os

# Delete the CSV file if it exists
file_path = "/kaggle/working/sample_submission.csv"

if os.path.exists(file_path):
    os.remove(file_path)  # <-- call the function with the file path
    print("🗑️ File deleted successfully.")
else:
    print("⚠️ File not found.")