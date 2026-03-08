import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("data/processed/processed_data.csv")

X=df.drop(["Label","Binary_Label"],axis=1)
y=df["Binary_Label"]

def log_transformation(X):   # Log Transforamtion
    skew_values = X.skew()
    highly_skewed=skew_values[skew_values>3]

    X_log = X.copy()

    for col in highly_skewed.index:
        if(X_log[col]<0).any():
            X_log[col]=X_log[col]-X_log[col].min()
    
        X_log[col] = np.log1p(X_log[col])

    return X_log

def feature_selection(X_log):  # Correlation based Feature Selection
    corr_matrix = X_log.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) 
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

    print("Highly correlated features:", len(to_drop))

    X_selected = X_log.drop(columns=to_drop)

    return X_selected

def feature_importance(X_selected):   #Random Forest Feature Importance
    sample_size = 200000   # 1 lakh rows
    sample = X_selected.sample(sample_size, random_state=42)
    y_sample = y.loc[sample.index]

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    rf.fit(sample,y_sample)

    importance = rf.feature_importances_
    feature_importance = pd.Series(importance, index=X_selected.columns).sort_values(ascending=False)

    top_features=feature_importance.head(30).index
    X_final=X_selected[top_features]    #final selected features

    return X_final

if __name__ == "__main__":

     X_log=log_transformation(X)
     X_selected=feature_selection(X_log)
     X_final=feature_importance(X_selected)

     print("X_final shape: ", X_final.shape)
     print("selected features: ",list(X_final.columns))

     final_df=X_final.copy()
     final_df["Binary_Label"]=y

     final_df.to_csv("data/processed/final_features.csv",index=False)
