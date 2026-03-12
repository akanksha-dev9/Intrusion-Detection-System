import pandas as pd
import joblib

# load model
model = joblib.load("models/rf_ids_model.pkl")

# load new data
df = pd.read_csv("data/processed/final_features.csv")

X = df.drop(["Binary_Label"], axis=1)
y = df["Binary_Label"]

# take some samples
X_sample = X.sample(5)

predictions = model.predict(X_sample)

print("Predictions of 5 samples:")

for i in predictions:
    if(i==0):
        print("Normal")
    else:
        print("Attack")
