# Generated from: part2house.ipynb
# Converted at: 2026-01-09T20:08:37.633Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # 🏠 House Prices: Advanced Regression Techniques (Part 2) .......................................................................................................![icons8-spaceship-64.png](attachment:05de71d4-66a5-4214-ba46-1f72a8a0cd5d.png)
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


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# 2. Load Dataset


df_test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
df_test.head()

# ## 3. Data Exploration & Cleaning
# 
# ## 3.1 Overview
# 
# **Check shape, missing values, data types.**


df_test.shape

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
df_test.isnull().sum()

df_test.info()

# ## 3.2 Visualization



# 1️⃣ Visualize missing data
sns.heatmap(df_test.isnull(), yticklabels=False, cbar=True, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()
missing_data = pd.DataFrame({
    'Missing_Values': df_test.isnull().sum(),
    'Percentage': (df_test.isnull().sum() / len(df_test)) * 100,
    'Data_Type': df_test.dtypes
})
# 3️⃣ Keep only columns with missing data
missing_data = missing_data[missing_data['Missing_Values'] > 0]

# 4️⃣ Sort by number of missing values (descending)
missing_data = missing_data.sort_values(by='Missing_Values', ascending=False)

# 5️⃣ Display the missing data summary
print("🔍 Missing Data Summary:\n")
display(missing_data)

df_test.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'FireplaceQu'], axis=1, inplace=True)

# ## 3.3 Filling missing values


# -------------------------
# Categorical columns (object) with less than 50% missing
# -------------------------
df_test['GarageCond'] = df_test['GarageCond'].fillna(df_test['GarageCond'].mode()[0])
df_test['GarageQual'] = df_test['GarageQual'].fillna(df_test['GarageQual'].mode()[0])
df_test['GarageFinish'] = df_test['GarageFinish'].fillna(df_test['GarageFinish'].mode()[0])
df_test['GarageType'] = df_test['GarageType'].fillna(df_test['GarageType'].mode()[0])
df_test['BsmtCond'] = df_test['BsmtCond'].fillna(df_test['BsmtCond'].mode()[0])
df_test['BsmtExposure'] = df_test['BsmtExposure'].fillna(df_test['BsmtExposure'].mode()[0])
df_test['BsmtQual'] = df_test['BsmtQual'].fillna(df_test['BsmtQual'].mode()[0])
df_test['BsmtFinType1'] = df_test['BsmtFinType1'].fillna(df_test['BsmtFinType1'].mode()[0])
df_test['BsmtFinType2'] = df_test['BsmtFinType2'].fillna(df_test['BsmtFinType2'].mode()[0])
df_test['MSZoning'] = df_test['MSZoning'].fillna(df_test['MSZoning'].mode()[0])
df_test['Functional'] = df_test['Functional'].fillna(df_test['Functional'].mode()[0])
df_test['Utilities'] = df_test['Utilities'].fillna(df_test['Utilities'].mode()[0])
df_test['KitchenQual'] = df_test['KitchenQual'].fillna(df_test['KitchenQual'].mode()[0])
df_test['Exterior2nd'] = df_test['Exterior2nd'].fillna(df_test['Exterior2nd'].mode()[0])
df_test['Exterior1st'] = df_test['Exterior1st'].fillna(df_test['Exterior1st'].mode()[0])
df_test['SaleType'] = df_test['SaleType'].fillna(df_test['SaleType'].mode()[0])

# -------------------------
# Numerical columns (float64) with less than 50% missing
# -------------------------
df_test['LotFrontage'] = df_test['LotFrontage'].fillna(df_test['LotFrontage'].median())
df_test['GarageYrBlt'] = df_test['GarageYrBlt'].fillna(df_test['GarageYrBlt'].median())
df_test['MasVnrArea'] = df_test['MasVnrArea'].fillna(df_test['MasVnrArea'].median())
df_test['BsmtFullBath'] = df_test['BsmtFullBath'].fillna(df_test['BsmtFullBath'].median())
df_test['BsmtHalfBath'] = df_test['BsmtHalfBath'].fillna(df_test['BsmtHalfBath'].median())
df_test['GarageCars'] = df_test['GarageCars'].fillna(df_test['GarageCars'].median())
df_test['GarageArea'] = df_test['GarageArea'].fillna(df_test['GarageArea'].median())
df_test['TotalBsmtSF'] = df_test['TotalBsmtSF'].fillna(df_test['TotalBsmtSF'].median())
df_test['BsmtUnfSF'] = df_test['BsmtUnfSF'].fillna(df_test['BsmtUnfSF'].median())
df_test['BsmtFinSF2'] = df_test['BsmtFinSF2'].fillna(df_test['BsmtFinSF2'].median())
df_test['BsmtFinSF1'] = df_test['BsmtFinSF1'].fillna(df_test['BsmtFinSF1'].median())


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
check_missing(df_test)


# # 3.4 Remove irrelevant columns


df_test.dropna(inplace=True)

df_test.shape

# # 4. Submission File


df_test.to_csv('Newdf_test.CSV',index=False)

# # Delete Submission File(If anything go wrong)


import os

# Delete the CSV file if it exists
file_path = "Newdf_test.CSV"

if os.path.exists(file_path):
    os.remove(file_path)
    print("🗑️ File deleted successfully.")
else:
    print("⚠️ File not found.")