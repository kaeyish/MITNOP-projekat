# -*- coding: utf-8 -*-
"""
Created on Mon May 26 22:25:08 2024

@author: Dušanka
"""
#%% importovi
import pandas as pd
import numpy as np
import os
import math
from pathlib import Path
from datetime import datetime
from dateutil.parser import parse
from mlxtend.frequent_patterns import apriori, association_rules

date_parser = lambda x: parse(x)

#%% ucitavanje podataka

path = Path(__file__).parent / "Crime.csv"

df = pd.read_csv(path)

print("Shape dataseta pre uklanjanja null: {}", df.shape)

if (pd.isna(df).values.any()):
    df = df.dropna()
    print("Uklonjene null vrednosti")
    
droppable = ['Occurred Through Date']
df = df.drop(droppable, axis = 1)

print("Shape dataseta posle uklanjanja null i end date: {}", df.shape)


df.rename({'Occurred From Date':'Ocurred Date'})

# print (df.dtypes)

# print (df["Reported Date"])



#%% parsiranje podataka


df['Occurred Time'] = df['Occurred From Date'].str.split(' ').str[1]

df['Occurred Date'] = df['Occurred From Date'].str.split(' ').str[0]
df['Occurred Day'] = df['Occurred Date'].str.split('-').str[2].astype(int)
df['Occurred Month'] = df['Occurred Date'].str.split('-').str[1].astype(int)
df['Occurred Year'] = df['Occurred Date'].str.split('-').str[0].astype(int)



temp = df["Reported Date"]

df.drop(["Reported Date"], axis = 1, inplace = True)

df['Reported Time'] = (temp.str.split(' ').str[1])
df['Reported Date'] = temp.str.split(' ').str[0].apply(date_parser)

df['Crime Type'] = df['Crime Type'].astype(str)
df['Case Number'] = df['Case Number'].astype(str)
df['Block Address'] = df['Block Address'].astype(str)

#df['Reported Time Parsed'] =  df['Reported Time'].apply(lambda a:  datetime.strptime(a, '%I:%M:%S:%f%p').time())


print ("Oblik dataseta posle razdvajanja kolona: {}",df.shape)
print (df.columns)

#%% 2024 CNT 

# Count the number of occurrences in the 'Occurred Year' column for the year 2024
count_2224 = df['Occurred Year'].value_counts().get(2022) + df['Occurred Year'].value_counts().get(2023) +df['Occurred Year'].value_counts().get(2024)  

count_2224_percent  = (count_2224 / df.shape[0])*100

print(f'The number of occurrences in the year 2024 is: {count_2224}; this is {count_2224_percent}% of data')


#%% Treniranje i priprema za isto
# training skup su podaci do 2022 sto je ~76%, refer to previous cell

train_data = df[df['Occurred Year'] < 2022]
test_data = df[df['Occurred Year'] >= 2022]

print(train_data.shape, test_data.shape)







#%% APRIORI WANNABE

# Kreiranje transakcija
basket = df.groupby(['Case Number', 'Crime Type']).size().unstack(fill_value=0)
basket = basket.applymap(lambda x: x > 0)  # konverzija u bool tip

# Primena Apriori algoritma
min_support_value = 0.01  
frequent_itemsets = apriori(basket, min_support=min_support_value, use_colnames=True)

# Generisanje pravila asocijacije
metric_type = "lift"  
min_threshold_value = 1
rules = association_rules(frequent_itemsets, metric=metric_type, min_threshold=min_threshold_value)

# Ispisivanje rezultata
print(frequent_itemsets)
print(rules)