import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# load the dataset
data = "Diabetes_cleaned.csv"
df = pd.read_csv(data)

#split the dataset into x and y
X = df.drop(columns=["Outcome", "Unnamed: 0"])
y = df["Outcome"]


#Train the model 
model = RandomForestClassifier()
model.fit(X,y)

#save the model
joblib.dump(model,'diabetes_app_new.pkl')
print("The model is saved sucessfully")
