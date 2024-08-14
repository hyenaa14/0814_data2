import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
data = pd.read_csv('./data/1.salary.csv',)
array= data.values
array.shape
X= array[:,0] #독립변수
Y= array[:,1] #종속변수

fig, ax=plt.subplots()
plt.clf()
plt.scatter(X,Y, label='random', color='gold',marker='*', s=30, alpha=0.5)
#근속연수*연봉
X1=X.reshape(-1,1)

#데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(X1,Y,test_size=0.2)

#모델 선택 및 학습
model=LinearRegression()
model.fit(X_train,  Y_train)

#예측
y_pred=model.predict(X_test)
print(y_pred)

plt.figure(figsize=(10,6))

plt.scatter(range(len(Y_test)), Y_test, color='blue', marker='o')
plt.plot(range(len(y_pred)), y_pred, color='r', marker='X')

plt.show()

mae=mean_absolute_error(y_pred, Y-test)
print(mae)