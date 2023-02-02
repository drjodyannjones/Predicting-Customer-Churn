# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 2023

@author: drjodyannjones
"""


# import libraries
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
import seaborn as sns
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        df = pd.read_csv(pth)
        # Encode target variable
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        return df
    except FileNotFoundError:
        print("We were not able to fine that file.")
    return


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # Separate categorical variables from numerical features
    categorical_features = [
        'Churn'
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    numeric_features = [
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
        'Avg_Utilization_Ratio'
    ]

    # Plot histograms for each numeric feature
    for col in numeric_features:
        df[col].hist()
        plt.savefig(os.path.join("./images/eda", f'{col}.png'),
                    box_inches='tight')

    # Plot value count for each categorical feature
    for col in categorical_features:
        df.col.value_counts('normalize').plot(kind='bar')
        plt.savefig(os.path.join("./images/eda", f'{col}.png'),
                    box_inches='tight')

    # Plot heatmap
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(os.path.join("./images/eda", 'correlation_matrix.png'),
                box_inches='tight')


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for category in category_lst:
        category_lst = []
        category_groups = df.groupby[category].mean()['Churn']
        for val in df[category]:
            category_lst.append(category_groups.loc[val])
        df["%s_%s" % (category, "Churn")] = category_lst
    return df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    categorical_columns = df.select_dtypes(include='object').columns.tolist()
    df = encoder_helper(df, categorical_columns, response='Churn')

    # Assign X and y variables
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

    # Split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=876)


def train_models():
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

    # Split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=876)

    # grid search
    rfc = RandomForestClassifier(random_state=876)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # scores
    print('random forest results')
    print('test results')
    print(classification_report(y_test, y_test_preds_rf))
    print('train results')
    print(classification_report(y_train, y_train_preds_rf))

    print('logistic regression results')
    print('test results')
    print(classification_report(y_test, y_test_preds_lr))
    print('train results')
    print(classification_report(y_train, y_train_preds_lr))


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    classification_report('Logistic Regression',
                          y_train,
                          y_test,
                          y_train_preds_lr,
                          y_test_preds_lr)
    plt.close()

    classification_report('Random Forest Regression',
                          y_train,
                          y_test,
                          y_train_preds_rf,
                          y_test_preds_rf)
    plt.close()


def feature_importance_plot(model, X_data, model_name, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
                    model: model object containing feature_importances_
                    X_data: pandas dataframe of X values
                    output_pth: path to store the figure
    output:
                     None
    '''

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title(f"Feature Importance for {model_name}")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save figure to output_pth
    fig_name = f'feature_importance_{model_name}.png'
    plt.savefig(os.path.join(output_pth, fig_name), bbox_inches='tight')

    # display feature importance figure
    plt.show()
    plt.close()


if __name__ == "__main__":
    # if this file is executed as a script,do this:
    import_data()
    perform_eda()
    perform_feature_engineering()
    train_models()
    classification_report_image()
    feature_importance_plot()
