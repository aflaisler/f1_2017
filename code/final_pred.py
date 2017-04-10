from __future__ import division
import urllib2
import numpy as np
import numpy
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import pylab as plot
import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics import regressionplots as smg

df = pd.read_csv('../pred.csv')

# Plot prediction save the graph with a timestamp
Y = np.array(df)
y_predicted, y, name = Y[:, 0], Y[:, 1], Y[:, 2]


def plot_pred(y_predicted, y, savefig=True):
    # y_predicted.reset_index(drop=1, inplace=1)
    index = np.argsort(y)
    y = y[index]
    # y.shape
    yhat = y_predicted[index]
    yy = pd.DataFrame([y, yhat])
    if yy.shape[1] > yy.shape[0]:
        yy = yy.T
    yy.reset_index(drop=0, inplace=1)
    plt.scatter(yy.index, yy[1], s=.4)
    plt.plot(yy.index, yy[1], ls='-', color='green', linewidth=.5)
    plt.plot(yy.index, yy[0], ls='-', color='red', linewidth=.5)
    if savefig:
        dt_ = datetime.datetime.now().strftime('%d%b%y_%H%M')
        plt.savefig("../graphs/" + dt_ + ".png")
    plt.show()

    plot_pred(y_predicted, y, savefig=True)


md = LinearRegression()
md.fit(xTrain, yTrain)

x = pd.DataFrame(xTrain)
y = yTrain
x['const'] = 1
prestige_model = sm.OLS(y, x).fit()

prestige_model.summary()

df_out.drop(['delta_race_qual'], 1, inplace=1)
y = df_out.pop('labelNormalized')
x = df_out

x['const'] = 1
prestige_model = sm.OLS(y, x).fit()
prestige_model.summary()

prestige_model
