import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_random_forest(df, target_column, test_size, random_state):
    """
    Train a Random Forest classifier.

    Parameters:
        df (DataFrame): Input dataframe.
        target_column (str): Name of the target column.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.

    Returns:
        RandomForestClassifier: Trained Random Forest classifier.
        float: Accuracy of the trained model.
    """
    # Split dataset into features and target variable
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

def train_logistic_regression(df, target_column, test_size, random_state):
    """
    Train a Logistic Regression classifier.

    Parameters:
        df (DataFrame): Input dataframe.
        target_column (str): Name of the target column.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.

    Returns:
        LogisticRegression: Trained Logistic Regression classifier.
        float: Accuracy of the trained model.
    """
    # Split dataset into features and target variable
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

def train_support_vector_machine(df, target_column, test_size, random_state):
    """
    Train a Support Vector Machine classifier.

    Parameters:
        df (DataFrame): Input dataframe.
        target_column (str): Name of the target column.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.

    Returns:
        SVC: Trained Support Vector Machine classifier.
        float: Accuracy of the trained model.
    """
    # Split dataset into features and target variable
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train the model
    model = SVC()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy
