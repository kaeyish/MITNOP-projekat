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


# Sredjivanje Occurred Time

df['Occurred Time'] = df['Occurred From Date'].str.split(' ').str[1]
df['Occurred Date'] = df['Occurred From Date'].str.split(' ').str[0]
df['Occurred Day'] = df['Occurred Date'].str.split('-').str[2].astype(int)
df['Occurred Month'] = df['Occurred Date'].str.split('-').str[1].astype(int)
df['Occurred Year'] = df['Occurred Date'].str.split('-').str[0].astype(int)


# Sredjivanje Reported Date

df['Reported Time'] = (df["Reported Date"].str.split(' ').str[1])
df['Reported Date'] = df["Reported Date"].str.split(' ').str[0].apply(date_parser)



# Sredjivanje Crime Type

# Split
#df['Crime Subtype'] = df['Crime Type'].str.split('-').str[1].str.strip().astype(str)
df['Crime Type'] = df['Crime Type'].str.split('-').str[0].str.strip().astype(str)

# Group
df['Crime Type'] = df['Crime Type'].replace('ASSAULTS (PRIOR TO SEPT 2018)', 'ASSAULT')
df['Crime Type'] = df['Crime Type'].replace('SHOPLIFT ROBBERY', 'SHOPLIFTING')
df['Crime Type'] = df['Crime Type'].replace('THEFT FROM COMMERCIAL BUILDING', 'THEFT')
df['Crime Type'] = df['Crime Type'].replace('THEFT OF AUTO PARTS', 'THEFT')
df['Crime Type'] = df['Crime Type'].replace('MOTOR VEHICLE THEFT', 'THEFT')



# Parsiranje Preostalih kolona

df['Case Number'] = df['Case Number'].astype(str)
df['Block Address'] = df['Block Address'].astype(str)
#df['Reported Time Parsed'] =  df['Reported Time'].apply(lambda a:  datetime.strptime(a, '%I:%M:%S:%f%p').time())



print ("Oblik dataseta posle razdvajanja kolona: {}",df.shape)

# Prebacivanje kolona za kasnije
cols = ['Case Number', 'Occurred From Date', 'Reported Date', 'Crime Type',
        'Block Address', 'Occurred Time', 'Occurred Date', 'Occurred Day',
        'Occurred Month', 'Occurred Year', 'Reported Time']

new_cols_order = ['Case Number', 'Occurred From Date', 'Reported Date', 'Crime Type',
                  'Block Address', 'Occurred Date', 'Occurred Time', 'Occurred Day',
                  'Occurred Month', 'Occurred Year', 'Reported Time']

df = df[new_cols_order]



print (df.columns)



#%% Mapiranje string podataka na integere

# Mapiranje tipova podataka
print("\nUnique values:\n", np.unique(df['Crime Type']))
print("TOTAL:", len(np.unique(df['Crime Type'])))

mapping_dict = {'ARSON': 1, 'ASSAULT': 2, 'BURGLARY': 3, 'DUI ARREST': 4, 
                'ROBBERY': 5, 'SHOPLIFTING': 6, 
                'THEFT': 7, 'VANDALISM': 8}

print ("********************")

# Apply the mapping to the dataset
df['Crime Type'] = df['Crime Type'].map(mapping_dict)

print("\nUnique values:\n", np.unique(df['Crime Type']))
print("TOTAL:", len(np.unique(df['Crime Type'])))


#%% 2024 CNT 

count_2224 = df['Occurred Year'].value_counts().get(2022) + df['Occurred Year'].value_counts().get(2023) +df['Occurred Year'].value_counts().get(2024)  

count_2224_percent  = (count_2224 / df.shape[0])*100

print(f'The number of occurrences in the year 2024 is: {count_2224}; this is {round(count_2224_percent,2)}% of data')

print (f'This makes Test:Train ratio approximately {round(100-count_2224_percent)}:{round(count_2224_percent)}')


#%% Razdvajanje na test i train skup
print("\n\nTraining skup su podaci do 2022 sto je ~76%, refer to previous cell for details!")
train_data = df[df['Occurred Year'] < 2022].iloc[:,3:4]
test_data = df[df['Occurred Year'] >= 2022].iloc[:,3:4]

print("Train and test shapes after split: \n\tTest",train_data.shape, "\n\tTrain", test_data.shape)
print("PRE RESHAPE: \n\n",train_data)

#%% Preprocessing za trening skup
# Preoblikovanje trening skupa iz 1d niza u 2d niz

dataset_train = train_data['Crime Type'].values 
dataset_train = np.reshape(dataset_train, (-1,1)) 
print("\n\nPOSLE RESHAPE: \n\n", dataset_train)


print("\nUnique values:\n", np.unique(dataset_train))

#%%
# Normalizacija trening skupa
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
# scaling dataset
scaled_train = scaler.fit_transform(dataset_train)
 
print("\n\n*****TRAIN*****\n",scaled_train[:5])

#%% Preprocessing za test skup

# Preoblikovanje test skupa iz 1d niza u 2d niz
dataset_test = test_data['Crime Type'].values 
dataset_test = np.reshape(dataset_test, (-1,1)) 
print("\n\nPOSLE RESHAPE: \n\n", dataset_test)


# Normalizacija test skupa
scaled_test = scaler.fit_transform(dataset_test)
 
print("\n\n*****TEST*****\n",scaled_test[:5])

#%% Izdvajanje x i y komponenti 

# Trening Skup
X_train = []
y_train = []
for i in range(50, len(scaled_train)):
	X_train.append(scaled_train[i-50:i, 0])
	y_train.append(scaled_train[i, 0])
	if i <= 51:
		print(X_train)
		print(y_train)
		print()


# Test Skup
X_test = []
y_test = []
for i in range(50, len(scaled_test)):
	X_test.append(scaled_test[i-50:i, 0])
	y_test.append(scaled_test[i, 0])

# %% Pretvaranje 2D niza u 3D niz koji je pogodan za RNN za x komponentu i 2D za y

# Train
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
y_train = np.reshape(y_train, (y_train.shape[0],1))
print("X_train :",X_train.shape,"y_train :",y_train.shape)

# Test
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
y_test = np.reshape(y_test, (y_test.shape[0],1))
print("X_test :",X_test.shape,"y_test :",y_test.shape)



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