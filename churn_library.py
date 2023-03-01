# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 2023

@author: drjodyannjones
"""

# Import Libraries

import logging
import os
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import shap  # model explainability
import joblib  # used to save the model
import pandas as pd  # data transformation
import numpy as np  # array computation
import matplotlib.pyplot as plt  # data visualization
import seaborn as sns
sns.set()  # data visualization

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create logger
logger = logging.getLogger(__name__)


# Model Preparation and Preprocessing

# Model Building

# Model Evaluation

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Import Data


def import_data(path):
    """
    returns dataframe for the csv found at pth

      input:
              pth: a path to the csv
      output:
              df: pandas dataframe
    """
    df = pd.read_csv(path)
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    df.drop(['Unnamed: 0', 'CLIENTNUM'], axis=1, inplace=True)
    return df

# EDA


def perform_eda(df):
    """
    perform eda on df and save figures to images folder
      input:
              df: pandas dataframe

      output:
              None
    """
    # Plot bar charts for categorical columns
    # This checks cardinality of data
    cat_columns = df.select_dtypes(include='object').columns.tolist()
    for col in cat_columns:
        plt.figure(figsize=(20, 10))
        df[col].value_counts().plot(kind='bar',
                                    title=f'{col} - %Churn')
        plt.savefig(os.path.join("./images/eda", f'{col}.png'))
        plt.close()

    # Plot histograms for numerical columns
    # This checks the distribution of data
    num_columns = df.select_dtypes(exclude='object').columns.tolist()
    for col in num_columns:
        plt.figure(figsize=(20, 10))
        df[col].plot(kind='hist',
                     title=f'{col} - Histogram')
        plt.savefig(os.path.join("./images/eda", f'{col}.png'))

        plt.close()

    # Plot correlation matrix
    plt.figure(figsize=(20, 10))
    sns.heatmap(df[num_columns].corr(),
                annot=True,
                square=True)
    plt.savefig(os.path.join("./images/eda", 'correlation_matrix.png'))

    return None

# Feature Engineering


def encoder_helper(df, category_list, response):
    '''
    helper function to turn each categorical column into a new column with
      proportion of churn for each category - associated with cell 15 from the notebook

      input:
              df: pandas dataframe
              category_lst: list of columns that contain categorical features
              response: string of response name

      output:
              df: pandas dataframe with new columns for
    '''
    # Create a list of categorical columns
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    for col in cat_columns:
        col_lst = []
        col_groups = df.groupby(col).mean(numeric_only=True)['Churn']

        for val in df[col]:
            col_lst.append(col_groups.loc[val])

        df[f'{col}_Churn'] = col_lst
    return df


def perform_feature_engineering(df, response):
    '''
    input:
                df: pandas dataframe
                response: string of response name

      output:
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
    '''
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X = df[keep_cols]
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=876)
    return X_train, X_test, y_train, y_test


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
   '''
    # Instantiate Random Forest Classifier
    random_forest_classifier = RandomForestClassifier(random_state=876)

    # Instantiate Logistic Regression Classifier
    logistic_reg_classifier = LogisticRegression(
        solver='lbfgs', max_iter=3000)

    # Setup parameter grid for grid search
    param_grid = {
        'n_estimators': [200, 500],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # Instantiate GridSearchCV for Random Forest Classifier
    grid_search_random_forest_classifier = GridSearchCV(
        estimator=random_forest_classifier, param_grid=param_grid, cv=5)
    # Fit Random Forest Classifier Model
    grid_search_random_forest_classifier.fit(X_train, y_train)

    # Fit Logistic Reg Classifier Model
    logistic_reg_classifier.fit(X_train, y_train)

    # Generate Predictions for Random Forest Classifier
    y_train_preds_rfc = grid_search_random_forest_classifier.best_estimator_.predict(
        X_train)
    y_test_preds_rfc = grid_search_random_forest_classifier.best_estimator_.predict(
        X_test)

    # Generate Predictions for Logistic Reg Classifier
    y_train_preds_lr = logistic_reg_classifier.predict(X_train)
    y_test_preds_lr = logistic_reg_classifier.predict(X_test)

    return None


def plot_classification_report(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rfc,
        y_test_preds_lr,
        y_test_preds_rfc):
    '''
    helper function that produces classification report for training and testing results and stores report as image in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rfc: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rfc: test predictions from random forest

    output:
            None
    '''
    pass

    return None


def classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rfc,
        y_test_preds_lr,
        y_test_preds_rfc):
    '''
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rfc: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rfc: test predictions from random forest

    output:
            None
    '''
    pass
    return None


def feature_importance_plot(model, X_data, output_path):
    '''
    creates and stores the feature importances in path
      input:
              model: model object containing feature_importances_
              X_data: pandas dataframe of X values
              output_pth: path to store the figure

      output:
               None
    '''
    return None


if __name__ == "__main__":
    # Load Data
    logger.info('Loading dataset...')
    df = import_data(r"./data/bank_data.csv")
    print(df.head())

    # Perform EDA
    logger.info('Performing EDA...')
    perform_eda(df)

    # Feature Engineering
    logger.info('Performing Feature Engineering...')
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    response = df['Churn']

    # Run encoder_helper function
    encoder_helper(df, category_list=cat_columns, response=response)

    # Perform feature engineering
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df, response)

    # Train Models
    logger.info('Training models...')
    train_models(X_train, X_test, y_train, y_test)
