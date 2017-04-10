
from __future__ import division
import urllib2
import numpy as np
from sklearn import datasets, linear_model
from math import sqrt
import matplotlib.pyplot as plot
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC, Lars
import seaborn as sns
sns.set(color_codes=True)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import datetime

md = Lars(fit_intercept=True, verbose=False, normalize=True, precompute='auto', n_nonzero_coefs=500,
          eps=2.2204460492503131e-16, copy_X=True, fit_path=True, positive=False)

md.fit(xList, labels)

mean_squared_error(labels, md.predict(xList))


index = np.argsort(labels)
y = labels[index]
y.shape
yhat = md.predict(xList)[index]


def plot_pred(y_predicted, y, savefig=True):
    index = np.argsort(y)
    y = y[index]
    # y.shape
    yhat = y_predicted[index]
    yy = pd.DataFrame([y, yhat])
    if yy.shape[1] > yy.shape[0]:
        yy = yy.T
    yy.reset_index(drop=0, inplace=1)
    plt.scatter(yy.index, yy[1], s=.4)
    plt.plot(yy.index, yy[0], ls='-', color='red', linewidth=.5)
    if savefig:
        dt_ = datetime.datetime.now().strftime('%d%b%y_%H%M')
        plt.savefig("./graphs/" + dt_ + ".png")
    plt.show()


plot_pred(md.predict(xList), labels)
