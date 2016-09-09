import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import numpy as np

data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv',index_col=0)
print data.head()
print data.shape

sns.pairplot(data,x_vars=['TV','Radio','Newspaper'],y_vars='Sales',size=7,aspect=0.7,kind='reg')
sns.plt.show()

feature_cols = ['TV','Radio','Newspaper']
X = data[feature_cols]
y = data['Sales'] # ou y.data.Sales

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)

lm = LinearRegression()
lm.fit(X_train,y_train)


print zip (feature_cols,lm.coef_)


y_pred = lm.predict(X_test)

print np.sqrt(metrics.mean_squared_error(y_test,y_pred))


#excluding Newspaper

feature_cols = ['TV','Radio']
X = data[feature_cols]
y = data['Sales'] # ou y.data.Sales

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)

lm = LinearRegression()
lm.fit(X_train,y_train)

print zip (feature_cols,lm.coef_)

y_pred = lm.predict(X_test)

print np.sqrt(metrics.mean_squared_error(y_test,y_pred))



