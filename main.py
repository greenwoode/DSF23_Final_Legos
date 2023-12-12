
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression

from sklearn import preprocessing

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures


from sklearn.decomposition import PCA
from scipy import stats

RAND_STATE = 0

legos = pd.read_csv('./data/setsPf.csv')
legos = legos[legos['Pieces']>=10]

y = legos['USD_MSRP']
#print(y)

X = legos.iloc[:,3:-1]

X['Theme'] = pd.factorize(X['Theme'])[0]
X['Theme_Group'] = pd.factorize(X['Theme_Group'])[0]
X['Subtheme'] = pd.factorize(X['Subtheme'])[0]
X['Category'] = pd.factorize(X['Category'])[0]
X['Packaging'] = pd.factorize(X['Packaging'])[0]
X['Availability'] = pd.factorize(X['Availability'])[0]
#print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RAND_STATE)

linModel = LinearRegression().fit(X_train, y_train)

linScore = linModel.score(X_test, y_test)
print(f'linScore:{round(linScore*100, 1)}%')
dump(linScore, 'linearModel.joblib')

poly1 = PolynomialFeatures(2)
Xp_train = poly1.fit_transform(X_train)

poly2 = PolynomialFeatures(2)
Xp_test = poly2.fit_transform(X_test)

polyModel = LinearRegression().fit(Xp_train, y_train)

polyScore = polyModel.score(Xp_test, y_test)
print(f'polyScore:{round(polyScore*100, 1)}%')
dump(polyScore, 'polyModel.joblib')

SGDModel = make_pipeline(StandardScaler(),SGDRegressor(random_state=RAND_STATE)).fit(X_train, y_train)

SGDScore = SGDModel.score(X_test, y_test)
print(f'SGDScore:{round(SGDScore*100, 1)}%')
dump(SGDScore, 'SGDModel.joblib')

BayesianModel = BayesianRidge().fit(X_train, y_train)

BayesianScore = BayesianModel.score(X_test, y_test)
print(f'BayesianScore:{round(BayesianScore*100, 1)}%')
dump(BayesianScore, 'BayesianModel.joblib')

pca = PCA().fit(X)
#print(f'First two PCS:{pca.explained_variance_ratio_[0:2]}')

PCs = pd.DataFrame(pca.components_,columns=X.columns, index=['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10'])
PCs['Explained Variance'] = pca.explained_variance_ratio_
print(f'{PCs}')

PDR = pd.DataFrame()
PDR['Pieces'] = X['Pieces']
PDR['Cost'] = y
PDR['Pieces/$'] = X['Pieces']/y

#print(PDR)
#print(legos)

PDR2 = PDR

q_low = PDR2['Pieces/$'].quantile(0.025)
q_hi  = PDR2['Pieces/$'].quantile(0.975)

PDR2 = PDR2[(PDR2['Pieces/$'] < q_hi) & (PDR2['Pieces/$'] > q_low)]

#PDR2 = PDR2[(np.abs(stats.zscore(PDR2['Pieces/$'])) < 3)]

print(f'Average pieces per dollar: {round(PDR["Pieces/$"].mean(), 2)}')
print(f'Average pieces per dollar 95%: {round(PDR2["Pieces/$"].mean(), 2)}')

print(f'Variance of pieces per dollar 95%: {round(PDR2["Pieces/$"].var(), 2)}')
print(f'Variance of pieces per dollar: {round(PDR["Pieces/$"].var(), 2)}')

#print(PDR2)

import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit
b, m = polyfit(X['Pieces'], y, 1)

plt.figure()
plt.title("Pieces vs Cost")
plt.xlabel("# of Pieces")
plt.ylabel("Price ($)")
plt.scatter(X['Pieces'], y, c='black', s=5, alpha=0.5)
plt.plot(X['Pieces'], b + m * X['Pieces'], '-', c='red')
plt.show()