# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 08:08:57 2019

@author: N827941
"""

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

ratings = pd.read_csv('ml-100k/u.data', sep='\t', names = ['userid', 'itemid', 'rating', 'timestamp'])
movies = pd.read_csv('ml-100k/u.item', sep='|', encoding = 'latin1', names= ['movieid', 'movietitle', 'releasedate', 'videoreleasedate',
              'IMDbURL', 'unknown', 'Action', 'Adventure', 'Animation',
              'Children''s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
              'Thriller', 'War', 'Western'])
users = pd.read_csv('ml-100k/u.user', sep='|', names=['userid', 'age', 'gender', 'occupation', 'zipcode'])

x = ratings.rating.value_counts().index
y = [ratings['rating'].value_counts()[i]/1000 for i in x]

plt.bar(x, y, align = 'center', color = 'lightblue', edgecolor = 'black', alpha = 0.7)
plt.xlabel('Stars')
plt.ylabel('Count (in thousands)')
plt.title('Rating')

sns.distplot(users.age)

movies['release_year'] = movies['releasedate'].str.split('-', expand = True)[2]
movies['release_year'] = movies['release_year'].fillna(value = movies['release_year'].median())
movies['release_year'] = movies.release_year.astype(int)
plt.figure(figsize = (20, 6))
sns.distplot(movies.release_year)    

sns.countplot(users.gender, palette = ['lightblue', 'pink'])

sns.countplot(users.occupation, color = 'lightblue')
plt.yticks(rotation = 0)
plt.xticks(rotation = 90)

genre_by_year = movies.groupby('release_year').sum()
genre_by_year = movies.drop(columns = ['movieid','movietitle', 'releasedate', 'videoreleasedate', 'IMDbURL']).T
genre_by_year


plt.figure(figsize = (20,7))
sns.heatmap(genre_by_year, cmap='YlGnBu')


items = ratings.groupby('itemid').count()
items = items[ratings.groupby('itemid').count()['userid'] > 100].index

items = ratings.loc[ratings.itemid.isin(items)]
items = items.groupby('itemid').mean()
items = items.sort_values('rating', ascending = False)
order = items.index

rating_list = items.rating[0:25]
items = movies.loc[movies['movieid'].isin(order)]
top_25_movies = items.set_index('movieid').loc[order]
top_25_movies = top_25_movies.iloc[0:25, 0]
top_25_movies = top_25_movies.reset_index()
top_25_movies['avg_rating'] = rating_list.values

result = pd.merge(ratings,users,how='inner',on='userid')
movies.rename(columns = {'movieid' : 'itemid'}, inplace = True)
result = pd.merge(result, movies, how='inner', on='itemid')

Genre_by_gender = result.groupby('gender').sum().loc[:, 'Action': 'Western']
Genre_by_gender['total'] = Genre_by_gender.sum(axis = 1)
Genre_by_gender.div(Genre_by_gender.total, axis = 0) * 100






