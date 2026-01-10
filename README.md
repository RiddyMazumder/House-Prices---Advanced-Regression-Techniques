## 🏡 House Prices: Advanced Regression Techniques
![House Prices Banner](https://github.com/RiddyMazumder/House-Prices---Advanced-Regression-Techniques/blob/main/kaggle_5407_media_housesbanner.png)
Predicting house prices is about more than just counting bedrooms or imagining a white picket fence. This project dives deep into the features that truly influence residential property values in Ames, Iowa. Using a dataset with 79 explanatory variables, this competition challenges us to predict the final sale price of each home with precision.

| 📌 **Project Overview**                                                                                                                                                                                                                       | 🏆 **Kaggle Competition & Score**                                                                                                                                                                                                                                                                  |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 📝 An end-to-end machine learning project covering **EDA**, **data preprocessing**, **feature engineering**, **model training**, and **evaluation** using **79 features of residential homes in Ames, Iowa** to predict the final sale price. | 🚀 **House Prices – Advanced Regression Techniques** <br> 🔗 [https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) <br> 📊 **Public Score:** *Add your score here (e.g., 0.145 RMSE)* |

## 🛠️ Tools & Libraries
| 🔧 **Category**              | 🛠️ **Libraries / Tools**                                                   |
| ---------------------------- | --------------------------------------------------------------------------- |
| Data Manipulation & Analysis | `pandas`, `numpy`                                                           |
| Data Visualization           | `matplotlib`, `seaborn`                                                     |
| Machine Learning             | `xgboost`, `sklearn` (`train_test_split`, `mean_squared_error`, `r2_score`) |
| Development                  | `Jupyter Notebook`, `.py` scripts, `.html` exports                          |
## 📂 Project Files
| File               | Description                                                               |
| ------------------ | ------------------------------------------------------------------------- |
| `part1house.ipynb` | Notebook covering data exploration, cleaning, and initial model building. |
| `part1house.py`    | Python script version of part 1 notebook.                                 |
| `part1house.html`  | HTML export of part 1 notebook for easy sharing.                          |
| `part2house.ipynb` | Notebook with advanced modeling, hyperparameter tuning, and analysis.     |
| `part2house.py`    | Python script version of part 2 notebook.                                 |
| `part2house.html`  | HTML export of part 2 notebook for easy sharing.                          |
| `README.md`        | Project documentation and overview.                                       |
## 🔍 Workflow & Methodology
| Step                               | Description                                                                                                   |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **1. Load Dataset**                | Load training and test datasets using `pandas`.                                                               |
| **2. Data Exploration & Cleaning** | Explore missing values, distributions, correlations, and outliers. Visualize with `seaborn` and `matplotlib`. |
| **3. Feature Engineering**         | Encode categorical variables, create new features, and handle skewed data.                                    |
| **4. Model Building**              | Split data into train/test sets and train **XGBoost Regressor**.                                              |
| **5. Model Evaluation**            | Evaluate performance using **RMSE** and **R² score**. Visualize residuals and feature importance.             |
| **6. Kaggle Submission**           | Generate predictions for test data and prepare a submission file for Kaggle.                                  |
| **7. Conclusion & Insights**       | Summarize key features influencing house prices and discuss possible improvements.                            |
## 📈 Model Performance
| Metric         | Description                                                    |
| -------------- | -------------------------------------------------------------- |
| Algorithm      | XGBoost Regressor                                              |
| Evaluation     | Root Mean Squared Error (RMSE), R² Score                       |
| Visualizations | Feature importance, prediction vs actual plots, residual plots |
| Goal           | High predictive accuracy for sale prices on unseen test data   |
## 🔮 Future Improvements
| Improvement           | Description                                                              |
| --------------------- | ------------------------------------------------------------------------ |
| Ensemble Methods      | Explore stacking or blending multiple models for better predictions.     |
| Hyperparameter Tuning | Use `GridSearchCV` or `Optuna` for optimal model parameters.             |
| Transformations       | Apply log-transformations to skewed target variables.                    |
| External Features     | Incorporate neighborhood and location-based data to enhance predictions. |
## 📖 References
| Resource                      | Link                                                                                                                                                               |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Kaggle Competition            | [https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) |
| XGBoost Documentation         | [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)                                                                                                 |
| Seaborn Visualization Library | [https://seaborn.pydata.org/](https://seaborn.pydata.org/)                                                                                                         |

