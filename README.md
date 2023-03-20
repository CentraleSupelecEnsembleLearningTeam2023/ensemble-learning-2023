# New York City Airbnb Price Prediction

## Authors
The project is done by [Karim El Hage](https://github.com/karimelhage), [Ali Najem](https://github.com/najemali), [Annabelle Luo](https://github.com/annabelleluo), [Xiaoyan Hong](https://github.com/EmmaHongW), [Antoine Cloute](https://github.com/AntAI-Git)

<a href="https://github.com/annabelleluo/ensemble-learning-2023/graphs/contributors"> 
  <img src="https://contrib.rocks/image?repo=annabelleluo/ensemble-learning-2023" />
</a>

## Introduction 
This is the course project for Ensemble Learning at CentraleSupélec, 2023. Airbnb has become a popular alternative to traditional hotels and has disrupted the hospitality industry. Through Airbnb, individuals can list their own properties as rental places. In New York City
alone, there are around 40,000 listings. While traditional hotels have teams that carefully measure
demand and supply to adjust pricing, as a host, it can be challenging to determine the optimal price
for a listing. The variation in types of listings can also make it difficult for renters to get an accurate
sense of fair pricing. In this project, we will use ensemble learning methods to predict the price of
Airbnb listings in New York City. 

## Data 
The data set for this project is obtained from Kaggle and contains the listings in New York City in
2019. The data set includes 15 features on listings, including:
- Name of the listing
- Neighborhood
- Price
- Review information
- Availability

The data set contains around 47,000 listings. You can find it [here](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data).

## Project Tasks
- Pre-processing and EDA
- Apply all approaches taught in the course and practiced in lab sessions (Decision Trees, Bagging,
Random forests, Boosting, Gradient Boosted Trees, AdaBoost, etc.) on this data set. The goal
is to predict the target variable: price of the listing.
- Compare performances of models using various metrics learned in class.

## Contents
- EDA - Exploratory Data Analysis notebook containing data distirbutions, correlation expploration, heatmaps and other key visualizations
- Feature Engineering - Features engineered for use in modeling (not all made it to the final processing)
- Preprocessing - A comprehensive list of functions that will be used for preprocessing
- Modeling - A comprehensive list of functions used for modeling training and evaluation
- Training - Conducted with main.py using aforementioned steps and using log(price) as the target variable. When vectorize_text was True, training was done on GPU. It is not recommened to run full benchmark argument when this parameter is True in preprocess_data.
- Evaluation - Metrics are exported to to a csv file and achieving an R^2 on the test set of 65.6% using log(price) as the evaluation target. Using price the evaluation target, the best RMSE on test set is 215 and best R^2 is 20.3% on test set.
- Re-evaluation (no outliers) - A branch no_outliers was created to re-evaluate the performance without price outliers. Using price as the evaluation target, the best R^2 on test set was 61.5% and best RMSE of  42.2.

## Recommendations for use of file - Important
- All parameters should only be modefied under if __name__ == "__main__": in main.py
- If the vectorize_text functionality is kept true in main.py during the definition of the features and targets using preprocess_data, expect long run time. train_random_forest expected to delay runtime. It is recommended to run xgboost on 'cuda' for this functionality. It is not recommended to run train_ensemble_models with this functionality due to runtime. It is recomended for train_ensemble models to have vectorize_text False in preprocess_data.
- Grid search for XGBoost and CatBoost not added to modeling.py due to heavy runtime. User should implement gridsearch for these seperately.
- The hyperparameters displayed for all training models in main.py were found on gridsearch based on the  of preprocessing shown. If parameters in preprocess_data function are altered hyperparameters may not perform optimally.
- A secondary branch "without-outlier" can be run to simulate the performance of the model without price outliers.




