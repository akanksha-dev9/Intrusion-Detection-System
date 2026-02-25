import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load(file_name):
    df=pd.read_csv(file_name)
    # print(df['Label'].value_counts())
    return df

def remove_missing_values(df):
    
    df.replace([np.inf,-np.inf], np.nan,inplace=True) #Replace infinite values with NaN

    missing=df.isnull().sum() #Find the missing values per column
    missing=missing[missing>0]
    missing=missing.sort_values(ascending=False)

    missing_percentage=(missing/len(df))*100 #find the missing percentage for each column

    high_missing=missing_percentage[missing_percentage>50].index #if missing percentage is very high for a particular column, its better to remove that column
    df.drop(columns=high_missing,inplace=True)

    moderate_missing=missing_percentage[(missing_percentage>=2) & (missing_percentage<=50)].index  #if neither high nor low so better to fill with some value
    for col in moderate_missing:
        df[col]=df[col].fillna(df[col].median())
    
    low_missing=missing_percentage[missing_percentage<2].index #if it is very low for certain column so better to remove that data sample
    df.dropna(subset=low_missing,inplace=True)

    return df

def remove_duplicates(df):
    df.drop_duplicates(inplace=True)
    return df

def remove_constant_columns(df):
    constant_col=df.columns[df.nunique()==1]
    df.drop(columns=constant_col,inplace=True)
    return df

def label_encoding(df):
    le=LabelEncoder()
    le.fit(df['Label'])
    df['Label']= le.transform(df['Label'])
    print(le.classes_)
    return df,le

def save_processed_file(df):
    df.to_csv("data/processed/wednesday_cleaned.csv", index=False) # save processed file
    

file_name="data/raw/Wednesday-workingHours.pcap_ISCX.csv"
df=load(file_name)
df.columns=df.columns.str.strip()

df=remove_missing_values(df)  #Remove missing Values
df=remove_duplicates(df)  #Remove duplicates data samples(rows)
df=remove_constant_columns(df)
df,le=label_encoding(df)  #Encoding
save_processed_file(df)

X=df.drop("Label", axis=1)
y=df['Label']

print(X.shape)
print(y.shape)


