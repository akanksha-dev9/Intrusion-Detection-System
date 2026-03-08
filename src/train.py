import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


def Train_test_split(X,y):  # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("X_train:",X_train.shape)
    print("X_test:",X_test.shape)

    print("y_train:",y_train.shape)
    print("y_test:",y_test.shape)

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


if __name__=="__main__":
    df=pd.read_csv("data/processed/final_features.csv")

    X=df.drop(["Binary_Label"],axis=1)
    y=df["Binary_Label"]

    X_train,X_test,y_train,y_test=Train_test_split(X,y)
    X_train_res,y_train_res=SMOTE_handle_imbalance(X_train,y_train)
    X_train_scaled,X_test_scaled=feature_scaling(X_train_res,X_test)
