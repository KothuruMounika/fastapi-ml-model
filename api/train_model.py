import pandas as pd
from sklearn.model_selection import train_test_split#splits our input into testing and training
from sklearn.linear_model import LogisticRegression#for importing logistic regression
import joblib
data={
    'math':[78,65,78,45,30,60,80,40,20,50],
    'science':[20,68,45,82,63,81,20,40,60,90],
    'English':[30,70,45,65,98,42,30,30,40,60],
    'Result':['Fail','Pass','Fail','Fail','Fail','Pass','Fail','Fail','Fail','Pass']
}
df=pd.DataFrame(data)
print(df)
df['Result']=df['Result'].map({'Pass':1,'Fail':0})
#train the model 
x=df[['math','science','English']]#if we assign multiple values take double brackets
y=df['Result']#if we assign only single value take single bracket
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#70%i/p  #30%i/p  #70%o/p  #30%o/p
#Call the model/algorithm Logistic Regression:
model=LogisticRegression()
model.fit(x_train,y_train)
joblib.dump(model,'model.pkl')


