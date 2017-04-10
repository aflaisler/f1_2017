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
pd.set_option('display.max_columns', None)


# Load tables into a python dictionary as dataframe

def mysql_into_df(db_connection):

    df_tables = pd.read_sql('show tables;', con=db_connection)
    type(df_tables)
    d = {}
    col_name = df_tables.columns[0]
    for table in df_tables[col_name]:
        # print table
        key = table
        value = pd.read_sql('SELECT * FROM ' + table + ';', con=db_connection)
        d[key] = value
    return d


# Qualification table is loaded separatedly to infare the timestamp

def load_qualif_data(d, db_connection):
    d['qualifying'] = pd.read_sql(
        'SELECT * FROM qualifying;', con=db_connection,
        parse_dates=['q1', 'q2', 'q3'])
    d['qualifying']['q1_sum'] = d['qualifying'].q1.dt.second + \
        d['qualifying'].q1.dt.minute * 60
    d['qualifying']['q2_sum'] = d['qualifying'].q2.dt.second + \
        d['qualifying'].q2.dt.minute * 60
    d['qualifying']['q3_sum'] = d['qualifying'].q3.dt.second + \
        d['qualifying'].q3.dt.minute * 60
    # d['qualifying'].fillna(0, inplace=1)
    d['qualifying'].drop(['q1', 'q2', 'q3'], 1, inplace=1)
    df_qual_1 = d['qualifying'].merge(d['drivers'].drop(
        ['number', 'url'], 1), on='driverId', how='left')
    # df_qual_1.shape
    df_qual_2 = df_qual_1.merge(d['races'].drop(
        ['url'], 1), on='raceId', how='left')
    df_qual_3 = df_qual_2.merge(d['constructors'][['constructorId', 'nationality']],
                                on='constructorId', how='left', suffixes=('', '_constructor'))
    df_qual_4 = df_qual_3
    # .merge(d['results'], on=['raceId', 'driverId'], how='left', suffixes=('', '_dfr'))
    df_qual_4['q_mean'] = df_qual_3[['q1_sum', 'q2_sum', 'q3_sum']].mean(1)
    df_qual_4['q_min'] = df_qual_3[['q1_sum', 'q2_sum', 'q3_sum']].min(1)
    df_qual_4.reset_index(drop=1, inplace=1)
    return df_qual_4


# Results table is loaded separatedly to infare the timestamp

def load_results_data(d, db_connection):
    d['results'] = pd.read_sql(
        'SELECT * FROM results;', con=db_connection,
        parse_dates=['fastestLapTime'])
    df_total_time = d['lapTimes'].groupby(['raceId', 'driverId'], as_index=False)[
        'milliseconds'].sum()
    df_min_lap_time = d['lapTimes'].groupby(['raceId', 'driverId'], as_index=False)[
        'milliseconds'].min()
    df_mean_lap_time = d['lapTimes'].groupby(['raceId', 'driverId'], as_index=False)[
        'milliseconds'].mean()
    df_results_1 = d['results'].merge(d['races'].drop(
        ['url', 'time'], 1), on='raceId', how='left')
    df_results_2 = df_results_1.merge(df_total_time[['raceId', 'driverId', 'milliseconds']], on=[
                                      'raceId', 'driverId'], how='left', suffixes=('', '_totalTime'))
    df_results_3 = df_results_2.merge(df_min_lap_time[['raceId', 'driverId', 'milliseconds']], on=[
                                      'raceId', 'driverId'], how='left', suffixes=('', '_bestLapTime'))

    df_results_4 = df_results_3.merge(df_mean_lap_time[['raceId', 'driverId', 'milliseconds']], on=[
                                      'raceId', 'driverId'], how='left', suffixes=('', '_meanLapTime'))
    df_results_4.reset_index(drop=1, inplace=1)
    return df_results_4


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


# Delete japan 2015 (no race data)

def del_japan15(df):
    df_out = df.loc[(df.year != 2015) & (df.circuitId != 22)]
    # print df_out.shape
    return df_out


# Create a unique ID to help merging Qualification and races data

def unique_id(df):
    df['id_'] = df.driverId.astype(
        str) + df.year.astype(str) + df.circuitId.astype(str)
    return df


# Merge Qualifications and races data and clean the results

def clean_data(d, con=db_connection):
    #
    df_qual = load_qualif_data(d, con)
    df_races = load_results_data(d, con)
    # remove Japan as no data for 2015 race
    df_qual = del_japan15(df_qual)
    df_races = del_japan15(df_races)
    # create unique id
    df_qual = unique_id(df_qual)
    df_races = unique_id(df_races)
    # merge the results
    df_out = df_races.merge(
        df_qual, on='id_', how='inner', suffixes=('', '_qual'))
    df_out = df_out[pd.isnull(df_out.q_min) == False]
    print df_out.shape
    return df_out.reset_index(drop=1), df_races.reset_index(drop=1), df_qual.reset_index(drop=1)


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


# Extract and format weather data

def wet_races(df_races, filename='../data/wet_races.csv'):
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
    df_wet = pd.concat([df_wet, pd.DataFrame(wet, columns=['wet_races'])], 1)
    return df_wet


#####################

def prepare_data_residuals(xList, residuals, df_wet, df_qual, names):
    # getting the residuals from initial model
    names = np.append(names, ['residuals'])
    df_residuals = pd.DataFrame(np.concatenate(
        [xList, pd.DataFrame(residuals)], 1), columns=names)
    # df_residuals = pd.DataFrame(residuals, columns=['residuals'])
    df_residuals.reset_index(drop=1, inplace=1)
    # merge weahter data
    # df_residuals_wi_weather = df_residuals.merge(
    # df_wet, how='outer', on=['year', 'round'])
    # merge circuit data
    df = pd.read_csv('../data/circuit_details.csv')
    df = pd.get_dummies(df, 'Type')
    df_out_circuit_details = pd.get_dummies(df, 'Direction')
    # add it back to the data
    df_residuals_wi_circuit_data = df_residuals.merge(
        df_out_circuit_details, how='left', on=['raceId'])
    # merge age of the drivers
    df_ = df_residuals_wi_circuit_data.reset_index(drop=1)
    col_ = np.array(['raceId', 'driverId', 'constructorId'], dtype=object)
    # df_qual_residuals = df_qual.loc[:, col_]
    # X = df_.merge(df_qual_residuals, how='left', on=['raceId', 'driverId'])
    # df_residuals_wi_weather.to_clipboard()
    # df_ = df_.loc[:, col_]
    X = df_.reset_index(drop=1)
    X.fillna(0, inplace=1)
    # xList = pd.DataFrame(X)
    X.drop(['q_min', 'raceId', 'Type_0'], axis=1, inplace=1)
    y_label = 'residuals'
    xList, names, labels = prep(X, label=y_label, split=False)
    np.shape(xList)
    return xList, names, labels


if __name__ == '__main__':
    # Connection to the MySQLdb dump from Ergast
    db_connection = MySQLdb.connect(host='localhost',
                                    user='root',
                                    db='f1_2016')

    db_connection = MySQLdb.connect(host='localhost',
                                    user='root',
                                    db='f1_2017')
    d = mysql_into_df(db_connection)
    df = load_results_data(d, db_connection=db_connection)
    df = df.merge(d['drivers'].loc[:, ['driverId', 'driverRef']],
                  how='left', on='driverId')
    # df.to_clipboard()
    # d.keys()
    # Merge Qualifications and Races data
    df_qual_and_race, df_races, df_qual = clean_data(d, db_connection)
    df_wet = wet_races(df_races, filename='../data/wet_races.csv')
    # Columns to keep
    y_label = 'milliseconds_bestLapTime'
    col_to_keep = ['position_qual', 'q_min', 'code', 'raceId', 'circuitId', 'driverId',
                   'year', 'name', 'positionOrder', 'round', 'dob', y_label]

    X_y = Xy_matrix(df_qual_and_race, col_to_keep, df_wet)

    xList, names, labels = prep(X_y, label=y_label, split=False)
    # Normalize the data to detect outliers
    xNormalized, labelNormalized, xList, labels = normalize(xList, labels)
    #
    # y=xNormalized[:,-1]-xNormalized[:,0]
    # yy=pd.DataFrame(y)
    # yy.to_clipboard()
    # yhat=df_all.delta_race_qual
    # yhat.to_clipboard()
    # np.shape(xList)
    # np.shape(labels)
    #
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
    #
    # # Residuals from full data set
    # X, xTest, y, yTest, RFmd = fastLapModel(xList, labels, names, full_set=1)
    # residuals = quality_residuals(
    #     RFmd, X, y, get_residuals=1, save_graph=0)
    # xList, labels = X, y
    # xList_2, names_2, labels_2 = prepare_data_residuals(
    #     X, residuals, df_wet, df_qual, names)
    # xTrain, xTest, yTrain, yTest, RFmd_2 = fastLapModel(
    #     xList_2, labels_2, names_2, multiple=0, full_set=0)
    #
    # quality_residuals(RFmd_2, xList_2, labels_2, get_residuals=0, save_graph=0)
