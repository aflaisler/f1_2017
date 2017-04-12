from __future__ import division
import MySQLdb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas_summary import DataFrameSummary
import json
import os
from datetime import timedelta
import datetime
from RFfastestLap import fastLapModel, quality_residuals, plot_importance
from f1_data import LoadF1Data
# pd.set_option('display.max_columns', None)


# Extract and format weather data

def wet_races(self, df_races, filename='../data/wet_races.csv'):
    df_wet = df_races.groupby(['year', 'round'], as_index=0)[
        'circuitId'].count()
    # df_wet.columns
    wet_data = pd.read_csv(filename, index_col='season')
    # fill the data for wet races
    # unique_raceId = df_wet['round'].unique()
    wet = []
    for i, j in np.array(df_wet.iloc[:, [0, 1]]):
        test = j in wet_data[wet_data.index == i].values
        wet.append(1) if test else wet.append(0)
    # len(wet)
    df_wet = pd.concat(
        [df_wet, pd.DataFrame(wet, columns=['wet_races'])], 1)
    return df_wet

# Prepare the data for the Regression


def prep(df, label, split=True):
    init = True
    xList = []
    labels = []
    names = []
    df.reset_index(drop=1, inplace=1)
    df_ = df.copy()
    labels = df_.pop(label)
    for row in range(df_.shape[0]):
        if init:
            names = df_.columns.values
            init = False
        row = df_.iloc[row]
        # convert row to floats
        floatRow = [num for num in row]
        xList.append(floatRow)
    if split:
        xList, X_test, labels, y_test = train_test_split(
            xList, labels, test_size=0.3)
        return xList, names, np.array(labels), X_test, np.array(y_test)
    else:
        return xList, names, np.array(labels)

# Normalize data to remove outliers (no need for prediction)


def normalize(xList, labels):
    if np.mean(labels) > 1:
        # calculate means and variances
        xMeans = np.mean(xList, axis=0)
        xSD = np.std(xList, axis=0)
        # use calculated mean and standard deviation to normalize xList
        xNormalized = (xList - xMeans) / xSD
        # Normalize labels
        meanLabel = np.mean(labels)
        sdLabel = np.std(labels)
        labelNormalized = (labels - meanLabel) / sdLabel
        return xNormalized, labelNormalized, xList, labels
    else:
        print 'the data has already been normalized'
        return xList, labels


# Remove outliers from the data (car that break down, etc.)


def rm_outliers1(xNormalized, labelNormalized, xList, labels, base_time=0, threshold=1):
    X_norm = pd.DataFrame(xNormalized)
    y_norm = pd.DataFrame(labelNormalized, columns=['labelNormalized'])
    X = pd.DataFrame(xList)
    y = pd.DataFrame(labels, columns=['labels'])
    df_norm = pd.concat([X_norm, y_norm], 1)
    df_ = pd.concat([X, y], 1)
    # df_.shape
    # df_norm.shape
    df_norm['delta_race_qual'] = abs(
        df_norm['labelNormalized'] - df_norm[base_time])
    df_out = df_norm.loc[df_norm.delta_race_qual < threshold]
    df_out.shape
    # avoid leakage with y-qualif
    df_ = df_.iloc[df_out.index, :]
    df_all = pd.concat([df_, df_out], 1)
    df_all.reset_index(drop=1, inplace=1)
    # y_norm_out = df_out.pop('labelNormalized')
    y_out = df_.pop('labels')
    # df_out.drop('delta_race_qual', axis=1, inplace=1)
    return df_all, df_out, np.array(df_), np.array(y_out)

# Create the data matrix


def Xy_matrix(df_qual_and_race, columns, df_wet):
    df_q_r_out = df_qual_and_race.loc[:, columns].reset_index(drop=1)
    df_q_r_out = df_q_r_out[(pd.isnull(
        df_q_r_out[y_label]) == False) & (pd.isnull(df_q_r_out.q_min) == False)].reset_index(drop=1)
    X = df_q_r_out.loc[:, ['q_min', 'position_qual', 'raceId', 'circuitId',
                           'driverId', 'year', 'round', 'dob', y_label]]
    # birth year / mo
    X['birth_year'] = map(lambda x: int(x.year), df_q_r_out['dob'])
    X['birth_mo'] = map(lambda x: int(x.month), df_q_r_out['dob'])
    X.drop('dob', axis=1, inplace=1)
    # adding wet as a feature
    # weather data
    df_races = d['races'].copy()
    # df_races.head()
    X = X.merge(df_wet.drop(['circuitId'], 1),
                how='left', on=['year', 'round'])
    # pit stop
    df_pits = d['pitStops'].groupby(['raceId', 'driverId'], as_index=0)[
        'milliseconds'].sum()
    df_pits.reset_index(drop=1, inplace=1)
    X_y = X.merge(df_pits, how='left', on=['raceId', 'driverId'])
    X_y.fillna(0, inplace=1)
    return X_y


if __name__ == '__main__':
    # load data 2016
    ld = LoadF1Data('f1_2016')
    df_qual_and_race, df_races, df_qual = ld.load()

    # Add weather data
    df_wet = wet_races(df_races, filename='../data/wet_races.csv')
    # Columns to keep
    y_label = 'milliseconds_bestLapTime'
    col_to_keep = ['position_qual', 'q_min', 'code', 'raceId', 'circuitId', 'driverId',
                   'year', 'name', 'positionOrder', 'round', 'dob', y_label]

    X_y = Xy_matrix(df_qual_and_race, col_to_keep, df_wet)

    xList, names, labels = prep(X_y, label=y_label, split=False)
    # Normalize the data to detect outliers
    xNormalized, labelNormalized, xList, labels = normalize(xList, labels)

    # Remove outliers
    df_all, df_out, xList, labels = rm_outliers1(
        xNormalized, labelNormalized, xList, labels)
    print "You are now good for the regression"
    # Fit model to the data
    xTrain, xTest, yTrain, yTest, RFmd = fastLapModel(
        xList, labels, names, full_set=1)
    # Plot predictions versus y and returns residuals
    quality_residuals(RFmd, xTest, yTest, get_residuals=0, save_graph=0)
    # Plot feature importance
    plot_importance(names, model=RFmd, savefig=1)
