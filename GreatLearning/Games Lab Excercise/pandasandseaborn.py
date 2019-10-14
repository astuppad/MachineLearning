# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 08:37:59 2019

@author: N827941
"""

import pandas as pd
import seaborn as sns

games = pd.read_csv('games.csv')

games.head()
games.info()
games['playingtime'].mean()

games[games['total_comments'] == games['total_comments'].max()]['name']
games[games['id'] == 1500]['name']
games[games['name'] == 'Zocken']['yearpublished']

games[games['total_comments'] == games['total_comments'].max()]['name']
games[games['total_comments'] == games['total_comments'].min()]['name']

games.groupby('type').mean()['minage']
games['id'].nunique()
games['type'].value_counts()
games[['playingtime', 'total_comments']].corr()

sns.set(color_codes=True)
games.dropna(inplace = True)
games.info()

sns.distplot(games['average_rating'])
sns.jointplot(games['minage'], games['average_rating'])
sns.pairplot(games[['playingtime', 'minage', 'average_rating']])
sns.stripplot(games['type'], games['playingtime'], jitter = True)