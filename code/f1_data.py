from __future__ import division
import MySQLdb
import pandas as pd
import numpy as np
import json
from datetime import timedelta
import datetime


class LoadF1Data(object):
    def __init__(self, database):
        self.database = database

    # create mySql Connection
    def mysql_connect(self):
        # Connection to the MySQLdb dump from Ergast
        self.db_connection = MySQLdb.connect(host='localhost',
                                             user='root',
                                             db=self.database)

    # Load tables into a python dictionary as dataframe
    def mysql_into_df(self):
        # first Extract all the table names
        # connect = self.db_connection
        df_tables = pd.read_sql('show tables;', self.db_connection)
        # store the tables in a dictionary
        d = {}
        col_name = df_tables.columns[0]
        # load individual table into the dictionary
        for table in df_tables[col_name]:
            key = table
            value = pd.read_sql('SELECT * FROM ' + table +
                                ';', self.db_connection)
            d[key] = value
        self.dictTables = d

    # Load qualification table separatedly to infare the timestamp
    def load_qualif_data(self):
        d = self.dictTables
        d['qualifying'] = pd.read_sql(
            'SELECT * FROM qualifying;', self.db_connection,
            parse_dates=['q1', 'q2', 'q3'])
        # Convert to milliseconds
        d['qualifying']['q1_sum'] = d['qualifying'].q1.dt.second + \
            d['qualifying'].q1.dt.minute * 60
        d['qualifying']['q2_sum'] = d['qualifying'].q2.dt.second + \
            d['qualifying'].q2.dt.minute * 60
        d['qualifying']['q3_sum'] = d['qualifying'].q3.dt.second + \
            d['qualifying'].q3.dt.minute * 60
        d['qualifying'].drop(['q1', 'q2', 'q3'], 1, inplace=1)
        # merge drivers data
        df_qual = d['qualifying'].merge(d['drivers'].drop(
            ['number', 'url'], 1), on='driverId', how='left')
        # merge races data
        df_qual = df_qual.merge(d['races'].drop(
            ['url'], 1), on='raceId', how='left')
        # add constructors data
        df_qual = df_qual.merge(d['constructors'][['constructorId', 'nationality']],
                                on='constructorId', how='left', suffixes=('', '_constructor'))
        df_qual.reset_index(drop=1, inplace=1)
        # add sum and mean for each qualification sessions
        df_qual['q_mean'] = df_qual[['q1_sum', 'q2_sum', 'q3_sum']].mean(1)
        df_qual['q_min'] = df_qual[['q1_sum', 'q2_sum', 'q3_sum']].min(1)
        return df_qual

    # Results table is loaded separatedly to infare the timestamp
    def load_results_data(self):
        d = self.dictTables
        d['results'] = pd.read_sql(
            'SELECT * FROM results;', self.db_connection,
            parse_dates=['fastestLapTime'])
        # Format total time to be added to results
        df_total_time = d['lapTimes'].groupby(['raceId', 'driverId'], as_index=False)[
            'milliseconds'].sum()
        df_min_lap_time = d['lapTimes'].groupby(['raceId', 'driverId'], as_index=False)[
            'milliseconds'].min()
        df_mean_lap_time = d['lapTimes'].groupby(['raceId', 'driverId'], as_index=False)[
            'milliseconds'].mean()
        # merge races data
        df_results = d['results'].merge(d['races'].drop(
            ['url', 'time'], 1), on='raceId', how='left')
        # merge details time
        df_results = df_results.merge(df_total_time[['raceId', 'driverId', 'milliseconds']], on=[
            'raceId', 'driverId'], how='left', suffixes=('', '_totalTime'))
        df_results = df_results.merge(df_min_lap_time[['raceId', 'driverId', 'milliseconds']], on=[
            'raceId', 'driverId'], how='left', suffixes=('', '_bestLapTime'))
        df_results = df_results.merge(df_mean_lap_time[['raceId', 'driverId', 'milliseconds']], on=[
            'raceId', 'driverId'], how='left', suffixes=('', '_meanLapTime'))
        df_results = df_results.merge(d['drivers'].loc[:, ['driverId', 'driverRef']],
                                      how='left', on='driverId')
        df_results.reset_index(drop=1, inplace=1)
        return df_results

    # Delete japan 2015 (no race data)
    def del_japan15(self, df):
        df_out = df.loc[(df.year != 2015) & (df.circuitId != 22)]
        # print df_out.shape
        return df_out

    # Create a unique ID to help merging Qualification and races data
    def unique_id(self, df):
        df['id_'] = df.driverId.astype(
            str) + df.year.astype(str) + df.circuitId.astype(str)
        return df

    # Merge Qualifications and races data and clean the results
    def clean_data(self):
        # load qualif and race data
        df_qual = self.load_qualif_data()
        df_races = self.load_results_data()
        # remove Japan as no data for 2015 race
        df_qual = self.del_japan15(df_qual)
        df_races = self.del_japan15(df_races)
        # create unique id
        df_qual = self.unique_id(df_qual)
        df_races = self.unique_id(df_races)
        # merge the results
        df_out = df_races.merge(
            df_qual, on='id_', how='inner', suffixes=('', '_qual'))
        df_out = df_out[pd.isnull(df_out.q_min) == False]
        print df_out.shape
        return df_out.reset_index(drop=1), df_races.reset_index(drop=1), df_qual.reset_index(drop=1)

    # load the data
    def load(self):
        self.mysql_connect()
        self.mysql_into_df()
        return self.clean_data()


if __name__ == '__main__':
    ld = LoadF1Data('f1_2016')
    df_qual_and_race, df_races, df_qual = ld.load()
    df_qual_and_race
