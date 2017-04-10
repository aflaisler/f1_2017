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

# RF model


def fastLapModel(xList, labels, names, multiple=0, full_set=0):
    X = numpy.array(xList)
    y = numpy.array(labels)
    featureNames = []
    featureNames = numpy.array(names)
    # take fixed holdout set 30% of data rows
    xTrain, xTest, yTrain, yTest = train_test_split(
        X, y, test_size=0.30, random_state=531)
    # for final model (no CV)
    if full_set:
        xTrain = X
        yTrain = y
    check_set(xTrain, xTest, yTrain, yTest)
    print "Fitting the model to the data set..."
    # train random forest at a range of ensemble sizes in order to see how the
    # mse changes
    mseOos = []
    m = 10 ** multiple
    nTreeList = range(500 * m, 1000 * m, 100 * m)
    # iTrees = 10000
    for iTrees in nTreeList:
        depth = None
        maxFeat = int(np.sqrt(np.shape(xTrain)[1])) + 1  # try tweaking
        RFmd = ensemble.RandomForestRegressor(n_estimators=iTrees, max_depth=depth, max_features=maxFeat,
                                              oob_score=False, random_state=531, n_jobs=-1)
        # RFmd.n_features = 5
        RFmd.fit(xTrain, yTrain)

        # Accumulate mse on test set
        prediction = RFmd.predict(xTest)
        mseOos.append(mean_squared_error(yTest, prediction))
    # plot training and test errors vs number of trees in ensemble
    plot.plot(nTreeList, mseOos)
    plot.xlabel('Number of Trees in Ensemble')
    plot.ylabel('Mean Squared Error')
    #plot.ylim([0.0, 1.1*max(mseOob)])
    plot.show()
    print("MSE")
    print(mseOos[-1])
    return xTrain, xTest, yTrain, yTest, RFmd


def quality_residuals(RFmd, X, y, get_residuals=0, save_graph=0):
    y_predicted, y = RFmd.predict(X), np.array(y)
    residuals = y_predicted - y
    plot_pred(y_predicted, y, save_graph)
    # for the second model
    return residuals if get_residuals else None


# normalize by max importance, this is not showing how and why feature are
# important to the model

def plot_importance(names, model, savefig=True):
    featureNames = numpy.array(names)
    featureImportance = model.feature_importances_
    featureImportance = featureImportance / featureImportance.max()
    sorted_idx = numpy.argsort(featureImportance)
    barPos = numpy.arange(sorted_idx.shape[0]) + .5
    plot.barh(barPos, featureImportance[sorted_idx], align='center')
    plot.yticks(barPos, featureNames[sorted_idx])
    plot.xlabel('Variable Importance')
    plot.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
    if savefig:
        dt_ = datetime.datetime.now().strftime('%d%b%y_%H%M')
        plt.savefig("../graphs/featureImportance_" + dt_ + ".png")
    plot.show()


# Plot prediction save the graph with a timestamp

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
    plt.plot(yy.index, yy[0], ls='-', color='red', linewidth=.5)
    if savefig:
        dt_ = datetime.datetime.now().strftime('%d%b%y_%H%M')
        plt.savefig("../graphs/" + dt_ + ".png")
    plt.show()


# Check the data before regression (no Na, size, etc)

def check_set(X_train, X_test, y_train, y_test):
    lst = [X_train, X_test, y_train, y_test]
    lst = [pd.DataFrame(elt) for elt in lst]
    df = pd.concat([lst[0], lst[2]], 1)
    bool_ = df.isnull().values.any()
    print 'NA present: %s' % (bool_)
    print 'Features: %s' % (lst[0].columns.values)
    print 'Label: %s' % (lst[2].columns.values)
    print [elt.shape for elt in lst]


if __name__ == '__main__':
    # Fit model to the data
    xTrain, xTest, yTrain, yTest, RFmd = fastLapModel(xList, labels, names)
    # Plot predictions versus y and returns residuals
    residuals = quality_residuals(RFmd, xTest, yTest, 0)
    # Plot feature importance
    plot_importance(names, model=RFmd)
