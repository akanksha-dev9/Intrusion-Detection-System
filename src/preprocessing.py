import numpy as np
import pandas as pd

def remove_missing_values(df):

    print("Handling missing values...")
    
    df.replace([np.inf,-np.inf], np.nan,inplace=True) #Replace infinite values with NaN

    missing=df.isnull().sum() #Find the missing values per column
    missing=missing[missing>0].sort_values(ascending=False)

    missing_percentage=(missing/len(df))*100 #find the missing percentage for each column

    #if missing percentage is very high for a particular column, its better to remove that column
    high_missing=missing_percentage[missing_percentage>50].index 
    df.drop(columns=high_missing,inplace=True)

    #if neither high nor low so better to fill with some value
    moderate_missing=missing_percentage[(missing_percentage>=2) & (missing_percentage<=50)].index  
    for col in moderate_missing:
        df[col]=df[col].fillna(df[col].median())
    
    #if it is very low for certain column so better to remove that data sample
    low_missing=missing_percentage[missing_percentage<2].index 
    df.dropna(subset=low_missing,inplace=True)

    return df

def remove_duplicates(df):
    print("Removing duplicates...")
    df.drop_duplicates(inplace=True)
    return df

def remove_constant_columns(df):
    print("Removing constant columns...")
    constant_col=df.columns[df.nunique()==1]
    df.drop(columns=constant_col,inplace=True)
    return df

def Binary_encoding(df):
    print("Binary Encoding for Normal/Attack as 0/1...")
    df['Binary_Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    return df

if __name__ == "__main__":

    files=[
        "data/raw/Wednesday-workingHours.pcap_ISCX.csv",
        "data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "data/raw/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "data/raw/Tuesday-WorkingHours.pcap_ISCX.csv",
        "data/raw/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "data/raw/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "data/raw/Friday-WorkingHours-Morning.pcap_ISCX.csv"
    ]

    dfs=[]

    for file in files:
        temp=pd.read_csv(file)
        temp.columns = temp.columns.str.strip()
        dfs.append(temp)

    df=pd.concat(dfs,axis=0,ignore_index=True)

    print("Before cleaning : ",df.shape)

    df=remove_missing_values(df)  #Remove missing Values
    df=remove_duplicates(df)  #Remove duplicates data samples(rows)
    df=remove_constant_columns(df) #Remove constant columns
    df=Binary_encoding(df) #Binary Encoding for Normal/Attack

    print("After cleaning : ",df.shape)

    #Memory Optimization
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')

    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')

    df.to_csv("data/processed/processed_data.csv",index=False)

    


