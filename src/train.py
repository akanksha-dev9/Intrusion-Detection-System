import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


def Train_test_split(X,y):  # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    return X_train,X_test,y_train,y_test

def SMOTE_handle_imbalance(X_train,y_train):  # SMOTE to handle imbalance classes
    smote = SMOTE(random_state=42)

    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("Training data shape",X_train_res.shape)
    print("Class distribution after applying SMOTE: ",y_train_res.value_counts())

    return X_train_res,y_train_res

def feature_scaling(X_train_res,X_test): # Feature Scaling
    scaler=StandardScaler()

    X_train_scaled=scaler.fit_transform(X_train_res)
    X_test_scaled=scaler.transform(X_test)

    return X_train_scaled,X_test_scaled

def train_logistic_regression(X_train_scaled, y_train_res, X_test_scaled, y_test):

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train_res)

    y_pred = model.predict(X_test_scaled)

    print("\nLogistic Regression Results")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    return model, accuracy_score(y_test, y_pred)

def train_random_forest(X_train_res, y_train_res, X_test, y_test):

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)

    print("\nRandom Forest Results")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    return model, accuracy_score(y_test, y_pred)

def train_xgboost(X_train_res, y_train_res, X_test, y_test):

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)

    print("\nXGBoost Results")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    return model, accuracy_score(y_test, y_pred)

if __name__=="__main__":
    df=pd.read_csv("data/processed/final_features.csv")

    X=df.drop(["Binary_Label"],axis=1)
    y=df["Binary_Label"]

    X_train,X_test,y_train,y_test=Train_test_split(X,y)
    X_train_res,y_train_res=SMOTE_handle_imbalance(X_train,y_train)
    X_train_scaled,X_test_scaled=feature_scaling(X_train_res,X_test)

    # Logistic Regression
    lr_model, lr_acc = train_logistic_regression(
        X_train_scaled, y_train_res, X_test_scaled, y_test
    )

    # Random Forest
    rf_model, rf_acc = train_random_forest(
        X_train_res, y_train_res, X_test, y_test
    )

    # XGBoost
    xgb_model, xgb_acc = train_xgboost(
        X_train_res, y_train_res, X_test, y_test
    )

    print("\nModel Comparison")
    print("Logistic Regression:", lr_acc)
    print("Random Forest:", rf_acc)
    # print("XGBoost:", xgb_acc)

    joblib.dump(rf_model, "models/rf_ids_model.pkl")
    print("Model saved successfully")