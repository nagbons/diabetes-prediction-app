import pandas as pd 
d1= pd.read_csv("Diabetes.csv")

d2= d1.loc[(d1['Glucose']!=0) & (d1['BloodPressure']!=0) &
     (d1['SkinThickness']!=0) & (d1['Insulin']!=0) & (d1['BMI']!=0)]

d1['Glucose'].replace(0,d2['Glucose'].mean(),inplace=True)
d1['BloodPressure'].replace(0,d2['BloodPressure'].mean(),inplace=True)
d1['SkinThickness'].replace(0,d2['SkinThickness'].mean(),inplace=True)
d1['Insulin'].replace(0,d2['Insulin'].mean(),inplace=True)
d1['BMI'].replace(0,d2['BMI'].mean(),inplace=True)

d1.to_csv("Diabetes_cleaned.csv")