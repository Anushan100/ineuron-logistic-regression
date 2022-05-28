import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn import metrics

df = sm.datasets.fair.load_pandas().data
df['affair'] = (df.affairs>0).astype(int)

x=df.drop(columns=["affair"])
y=df['affair']

logreg=LogisticRegression()

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=3,test_size=0.3)

logreg.fit(x_train,y_train)

pickle.dump(logreg,open('final_model.pkl','wb'))
load_model=pickle.load(open('final_model.pkl','rb'))