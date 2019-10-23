import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split 

df_raw = pd.read_json(r"data.json")
data = df_raw[['age','countGroups']]


X = np.array(data['age']).reshape(-1, 1) 
y = np.array(data['countGroups']).reshape(-1, 1) 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) 
  
regr = linear_model.LinearRegression() 
  
regr.fit(X_train, y_train) 
print(regr.score(X_test, y_test))

y_pred = regr.predict(X_test) 
plt.scatter(X_test, y_test, color ='b') 
plt.plot(X_test, y_pred, color ='k') 
  
plt.show()

