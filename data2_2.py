from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
data = pd.read_csv('./data/2.iris.csv',)

header= ['sepal-length', 'sepal-width','petal-length','petal width','class']
X= array[:,4]
Y= array[:,4]

fig, ax= plt.subplots()
plt.clf()
plt.scatter(X,Y, label='random', color='gold',marker='*', s=30, alpha=0.5)
