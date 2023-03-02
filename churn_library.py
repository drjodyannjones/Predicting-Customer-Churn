# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 2023

@author: drjodyannjones
"""

# Import Libraries

import logging
import os
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import normalize
# import shap  # model explainability
# import joblib  # used to save the model
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
              data_frame: pandas dataframe
    """
    data_frame = pd.read_csv(path)
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    data_frame.drop(['Unnamed: 0', 'CLIENTNUM'], axis=1, inplace=True)
    return data_frame

# EDA


def perform_eda(data_frame):
    """
    perform eda on data_frame and save figures to images folder
      input:
              data_frame: pandas dataframe

      output:
              None
    """
    # Plot bar charts for categorical columns
    # This checks cardinality of data
    cat_columns = data_frame.select_dtypes(include='object').columns.tolist()
    for col in cat_columns:
        plt.figure(figsize=(20, 10))
        data_frame[col].value_counts().plot(kind='bar',
                                            title=f'{col} - %Churn')
        plt.savefig(os.path.join("./images/eda", f'{col}.png'))
        plt.close()

    # Plot histograms for numerical columns
    # This checks the distribution of data
    num_columns = data_frame.select_dtypes(exclude='object').columns.tolist()
    for col in num_columns:
        plt.figure(figsize=(20, 10))
        data_frame[col].plot(kind='hist',
                             title=f'{col} - Histogram')
        plt.savefig(os.path.join("./images/eda", f'{col}.png'))

        plt.close()

    # Plot correlation matrix
    plt.figure(figsize=(20, 10))
    sns.heatmap(data_frame[num_columns].corr(),
                annot=True,
                square=True)
    plt.savefig(os.path.join("./images/eda", 'correlation_matrix.png'))


# Feature Engineering


def encoder_helper(data_frame, category_list, response):
    '''
    helper function to turn each categorical column into a new column with
      proportion of churn for each category - associated with cell 15 from the notebook

      input:
              data_frame: pandas dataframe
              category_lst: list of columns that contain categorical features
              response: string of response name

      output:
              data_frame: pandas dataframe with new columns for
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
        col_groups = data_frame.groupby(col).mean(numeric_only=True)['Churn']

        for val in data_frame[col]:
            col_lst.append(col_groups.loc[val])

        data_frame[f'{col}_Churn'] = col_lst
    return data_frame


def perform_feature_engineering(data_frame, response):
    '''
    input:
                data_frame: pandas dataframe
                response: string of response name

      output:
                x_train: x_variables training data
                x_test: x_variables testing data
                y_train: y_variable training data
                y_test: y_variable testing data
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

    x_variables = data_frame[keep_cols]
    y_variable = data_frame['Churn']
    x_train, x_test, y_train, y_test = train_test_split(x_variables,
                                                        y_variable,
                                                        test_size=0.2,
                                                        random_state=876)
    return x_train, x_test, y_train, y_test


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: x_variables training data
              x_test: x_variables testing data
              y_train: y_variable training data
              y_test: y_variable testing data
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
    grid_search_random_forest_classifier.fit(x_train, y_train)

    # Fit Logistic Reg Classifier Model
    logistic_reg_classifier.fit(x_train, y_train)

    # Generate Predictions for Random Forest Classifier
    y_train_preds_rfc = grid_search_random_forest_classifier.best_estimator_.predict(
        x_train)
    y_test_preds_rfc = grid_search_random_forest_classifier.best_estimator_.predict(
        x_test)

    # Generate Predictions for Logistic Reg Classifier
    y_train_preds_lr = logistic_reg_classifier.predict(x_train)
    y_test_preds_lr = logistic_reg_classifier.predict(x_test)


def plot_classification_report(model_name,
                               y_train,
                               y_test,
                               y_train_preds,
                               y_test_preds):
    '''
    helper function - produces classification report and saves to images folder
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
    plt.rc('figure', figsize=(5, 5))

    # Plot Classification report on Train dataset
    plt.text(0.01, 1.25,
             str(f'{model_name} Train'),
             {'fontsize': 10},
             fontproperties='monospace'
             )
    plt.text(0.01, 0.05,
             str(classification_report(y_train, y_train_preds)),
             {'fontsize': 10},
             fontproperties='monospace'
             )

    # Plot Classification report on Test dataset
    plt.text(0.01, 0.6,
             str(f'{model_name} Test'),
             {'fontsize': 10},
             fontproperties='monospace'
             )
    plt.text(0.01, 0.7,
             str(classification_report(y_test, y_test_preds)),
             {'fontsize': 10},
             fontproperties='monospace'
             )

    plt.axis('off')

    # Save figure to ./images folder
    fig_name = f'Classification_report_{model_name}.png'
    plt.savefig(
        os.path.join(
            "./images/results",
            fig_name),
        bbox_inches='tight')

    # Display figure
    plt.show()
    plt.close()


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
    plot_classification_report('Logistic Regression',
                               y_train,
                               y_test,
                               y_train_preds_lr,
                               y_test_preds_lr)
    plt.close()

    plot_classification_report('Random Forest',
                               y_train,
                               y_test,
                               y_train_preds_rfc,
                               y_test_preds_rfc)
    plt.close()


def feature_importance_plot(model, x_data, model_name, output_path):
    '''
    creates and stores the feature importances in path
      input:
              model: model object containing feature_importances_
              x_data: pandas dataframe of x_variables values
              output_pth: path to store the figure

      output:
               None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title(f"Feature Importance for {model_name}")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    # Save figure to output_pth
    fig_name = f'feature_importance_{model_name}.png'
    plt.savefig(os.path.join(output_path, fig_name), bbox_inches='tight')

    # display feature importance figure
    plt.show()
    plt.close()


if __name__ == "__main__":
    # Load Data
    logger.info('Loading dataset...')
    data_frame = import_data(r"./data/bank_data.csv")
    print(data_frame.head())

    # Perform EDA
    logger.info('Performing EDA...')
    perform_eda(data_frame)

    # Feature Engineering
    logger.info('Performing Feature Engineering...')
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    response = data_frame['Churn']

    # Run encoder_helper function
    encoder_helper(data_frame, category_list=cat_columns, response=response)

    # Perform feature engineering
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        data_frame, response)

    # Train Models
    logger.info('Training models...')
    train_models(x_train, x_test, y_train, y_test)
