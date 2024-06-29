# -*- coding: utf-8 -*-
"""
Created on Mon May 26 22:25:08 2024

@author: Dušanka
"""

#%% importovi
import pandas as pd
from pandas import concat
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from dateutil.parser import parse
from mlxtend.frequent_patterns import apriori, association_rules

date_parser = lambda x: parse(x)

#%% HOTENCODE FOO

def hotencode (df, colname):
    n  = np.unique(df[colname])
    idx = 1
    for i in n:
      df[colname] = df[colname].apply(lambda a: idx if a == i else a)    
      idx+=1

#%%

#descale foo
def trans(y):
    
    y = y.flatten()
    new_df = pd.DataFrame(columns=['Crime Type', 'Block Address' , 'Occurred Day', 'Occurred Month', 'Occurred Year'])
    for i in range(0,len(y),5):
        elements = y[i:i+5]
        line = {'Crime Type':elements[0],'Block Address': elements[1], 'Occurred Day': elements[2],'Occurred Month':elements[3], 'Occurred Year': elements[4]}
        new_row_df = pd.DataFrame([line])

        # Concatenate the new row DataFrame with the original DataFrame
        new_df = pd.concat([new_row_df, new_df], ignore_index=True)
    
    return new_df
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


df = df.rename({'Occurred From Date':'Ocurred Date'})

print (df.dtypes)

print (df["Case Number"])

df['Case Number'] = df['Case Number'].apply(lambda x: x.replace('-', ''))
df['Case Number'] = pd.to_numeric(df['Case Number'], errors='coerce')

print (df["Case Number"])

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

#%% mapiranje stringova na intove

hotencode(df, 'Block Address')
hotencode(df, "Crime Type")


print (df["Block Address"])

print (df.dtypes)

#%%

x = df[['Crime Type', 'Block Address' , 'Occurred Day', 'Occurred Month', 'Occurred Year']]
#y = df['Crime Type']

#%% 2024 CNT 

count_2224 = x['Occurred Year'].value_counts().get(2022) + x['Occurred Year'].value_counts().get(2023) +x['Occurred Year'].value_counts().get(2024)  

count_2224_percent  = (count_2224 / df.shape[0])*100

print(f'The number of occurrences in the year 2024 is: {count_2224}; this is {round(count_2224_percent,2)}% of data')

print (f'This makes Test:Train ratio approximately {round(100-count_2224_percent)}:{round(count_2224_percent)}')


#%% Razdvajanje na test i train skup
print("\n\nTraining skup su podaci do 2022 sto je ~76%, refer to previous cell for details!")
train_data = x[x['Occurred Year'] < 2022]
test_data = x[x['Occurred Year'] >= 2022]
print("Train and test shapes after split: \n\tTest",train_data.shape, "\n\tTrain", test_data.shape)
print("PRE RESHAPE: \n\n",train_data)

#%% Preprocessing za trening skup
# Preoblikovanje trening skupa iz 1d niza u 2d niz

dataset_train = train_data.values 
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
dataset_test = test_data.values 
dataset_test = np.reshape(dataset_test, (-1,1)) 


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

print (X_test.shape)


#%% Ucitavanje biblioteka za neuronsku mrezu

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras import metrics
from keras.layers import GRU, Bidirectional
from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn import metrics
from sklearn.metrics import mean_squared_error

#%% Treniranje 
# initializing the RNN
regressor = Sequential()

# adding RNN layers and dropout regularization
regressor.add(SimpleRNN(units = 50, 
						activation = "tanh",
						return_sequences = True,
						input_shape = (X_train.shape[1],1)))
regressor.add(Dropout(0.2))

regressor.add(SimpleRNN(units = 50, 
						activation = "tanh",
						return_sequences = True))

regressor.add(SimpleRNN(units = 50,
						activation = "tanh",
						return_sequences = True))

regressor.add(SimpleRNN(units = 50, 
						activation = "tanh",
						return_sequences = True))



regressor.add( SimpleRNN(units = 50))

# adding the output layer
regressor.add(Dense(units = 1,activation='sigmoid'))

# compiling RNN
regressor.compile(optimizer = SGD(learning_rate=0.01,
								momentum=0.9, 
								nesterov=True), 
				loss = "mean_squared_error")

# fitting the model
regressor.fit(X_train, y_train, epochs = 10, batch_size = 512)
regressor.summary()

#%% LSTM Model
#Initialising the model
regressorLSTM = Sequential()

#Adding LSTM layers


# Adding LSTM layers
regressorLSTM.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1],1)))
regressorLSTM.add(Dropout(0.3))  # Add dropout with 30% probability

regressorLSTM.add(LSTM(100, return_sequences=True))  # Increase number of LSTM units

# Adding Dense layers
regressorLSTM.add(Dense(50, kernel_regularizer=l2(0.01)))  # Add L2 regularization

regressorLSTM.add(LSTM(50, 
					return_sequences = False))

regressorLSTM.add(Dense(1))

#Compiling the model
regressorLSTM.compile(optimizer = 'adam',
					loss = 'mean_squared_error',
					metrics = ["accuracy"])

#Fitting the model
regressorLSTM.fit(X_train, 
				y_train, 
				batch_size = 512,   
				epochs = 10)
regressorLSTM.summary()
#%% GRU
 
#Initialising the model
regressorGRU = Sequential()
 
# GRU layers with Dropout regularisation
regressorGRU.add(GRU(units=50, 
                     return_sequences=True,
                     input_shape=(X_train.shape[1],1),
                     activation='tanh'))
regressorGRU.add(Dropout(0.2))
 
 
regressorGRU.add(GRU(units=50, 
                     activation='tanh'))
 
# The output layer
regressorGRU.add(Dense(units=1,
                       activation='relu'))
# Compiling the RNN
regressorGRU.compile(optimizer = 'adam',
					loss = 'mean_squared_error',
					metrics = ["accuracy"])
# Fitting the data
regressorGRU.fit(X_train,y_train,epochs=10,batch_size=512)
regressorGRU.summary()

#%% Testiranje
y_GRU = regressorGRU.predict(X_test)
y_RNN = regressor.predict(X_test)
y_LSTM = regressorLSTM.predict(X_test)


print (y_RNN.shape, y_LSTM.shape, y_GRU.shape)

#%% reverse scaler

y_GRU_O = scaler.inverse_transform(y_GRU)
y_RNN_O = scaler.inverse_transform(y_RNN) 
y_LSTM_O = scaler.inverse_transform(y_LSTM) 

print (y_LSTM_O[:10])

#%% transform back to numpy array

y_GRU_O = trans(y_GRU_O) 
y_RNN_O = trans(y_RNN_O) 
y_LSTM_O = trans(y_LSTM_O) 


#%%


#print (y_GRU_O)
print (y_LSTM_O[:10])
#print (y_RNN_O[:10])



#%% Vizualizacija

fig, axs = plt.subplots(3,figsize =(18,12),sharex=True, sharey=False)
fig.suptitle('Model Predictions')

#Simple RNN Plot
#axs[2].plot(train_data.index[150:], train_data['Crime Type'][150:], label = "train_data", color = "b")
axs[2].plot(test_data.index, test_data['Block Address'], label = "test_data", color = "g")
axs[2].plot(y_RNN_O.index, y_RNN_O['Block Address'], label = "y_RNN", color = "brown")

axs[2].legend()
axs[2].title.set_text("Basic RNN")

#Plot for LSTM predictions
#axs[0].plot(train_data.index[150:], train_data['Crime Type'][150:], label = "train_data", color = "b")
axs[0].plot(test_data.index, test_data['Block Address'], label = "test_data", color = "g")
axs[0].plot(y_LSTM_O.index, y_LSTM_O['Block Address'], label = "y_LSTM", color = "orange")
axs[0].legend()
axs[0].title.set_text("LSTM")

#Plot for GRU predictions
#axs[1].plot(train_data.index[150:], train_data['Crime Type'][150:], label = "train_data", color = "b")
axs[1].plot(test_data.index, test_data['Block Address'], label = "test_data", color = "g")
axs[1].plot(y_GRU_O.index, y_GRU_O['Block Address'], label = "y_GRU", color = "red")
axs[1].legend()
axs[1].title.set_text("GRU")

#%%
plt.xlabel("Days")
plt.ylabel("Crime Rate")

plt.show()



#%% APRIORI WANNABE

# Kreiranje transakcija

#Occurred month and crime type
basket = df.groupby(['Occurred Month','Block Address', 'Crime Type']).size().unstack(fill_value=0)
basket = basket.applymap(lambda x: x > 0)  # konverzija u bool tip

print (basket)

# Primena Apriori algoritma
min_support_value = 0.01  
frequent_itemsets = apriori(basket, min_support=min_support_value, use_colnames=True)
print(frequent_itemsets)

# Generisanje pravila asocijacije
metric_type = "lift"  
min_threshold_value = 1
rules = association_rules(frequent_itemsets, metric=metric_type, min_threshold=min_threshold_value)

print (rules)
# Ispisivanje rezultata
print(frequent_itemsets.sort_values(by='support', ascending=False).head(10))
#print(rules)

#%% VISUALIZATION
#Random sampling of association rules for comparison of confidence and lift values
rules_random=rules.sample(20, random_state = 42)
rules_lift = rules_random[['support']].to_numpy()
rules_lift = (rules_lift/rules_lift.max()).transpose()[0]
rules_conf = rules_random[['confidence']].to_numpy()
rules_conf = (rules_conf/rules_conf.max()).transpose()[0]
width = 0.40
plt.figure(figsize=(12, 6), dpi=200)

# plot data in grouped manner of bar type
plt.bar(np.arange(len(rules_random))-0.2,rules_lift, width, color='black')
plt.bar(np.arange(len(rules_random))+0.2,rules_conf, width, hatch='//', edgecolor='black', facecolor='white')
plt.xlabel('Instance index')
plt.ylabel('Normalized metric value')
plt.legend(['lift','confidence'])
plt.xticks(range(0,10));


#%%
#Block Address and Crime type
basket = df.groupby(['Block Address', 'Crime Type']).size().unstack(fill_value=0)
basket = basket.applymap(lambda x: x > 0)  # konverzija u bool tip

print (basket)

# Primena Apriori algoritma
min_support_value = 0.01  
frequent_itemsets = apriori(basket, min_support=min_support_value, use_colnames=True)
print(frequent_itemsets)

# Generisanje pravila asocijacije
metric_type = "lift"  
min_threshold_value = 1
rules = association_rules(frequent_itemsets, metric=metric_type, min_threshold=min_threshold_value)

print (rules)
# Ispisivanje rezultata
#print(frequent_itemsets)
#print(rules)

#%% CT LOC VIS
#Random sampling of association rules for comparison of confidence and lift values
rules_random=rules.sample(10, random_state = 42)
rules_lift = rules_random[['lift']].to_numpy()
rules_lift = (rules_lift/rules_lift.max()).transpose()[0]
rules_conf = rules_random[['confidence']].to_numpy()
rules_conf = (rules_conf/rules_conf.max()).transpose()[0]
width = 0.40
plt.figure(figsize=(12, 6), dpi=200)

# plot data in grouped manner of bar type
plt.bar(np.arange(len(rules_random))-0.2,rules_lift, width, color='black')
plt.bar(np.arange(len(rules_random))+0.2,rules_conf, width, hatch='//', edgecolor='black', facecolor='white')
plt.xlabel('Instance index')
plt.ylabel('Normalized metric value')
plt.legend(['lift','confidence'])
plt.xticks(range(0,10));

#%%

#Block Address and Occurred Date
basket = df.groupby(['Block Address', 'Occurred Date']).size().unstack(fill_value=0)
basket = basket.applymap(lambda x: x > 0)  # konverzija u bool tip

if (basket.size > 0):
    print(basket)
else: print ("Prazan!")

# Primena Apriori algoritma
min_support_value = 0.01  
frequent_itemsets = apriori(basket, min_support=min_support_value, use_colnames=True)
if (frequent_itemsets.size > 0):
    print(frequent_itemsets)
else: print ("Prazan!")

# Generisanje pravila asocijacije
metric_type = "lift"  
min_threshold_value = 1
rules = association_rules(frequent_itemsets, metric=metric_type, min_threshold=min_threshold_value)

if (rules.size > 0):
    print(rules)
else: print ("Prazan!")
# Ispisivanje rezultata
#print(frequent_itemsets)
#print(rules)

#%% CT LOC VIS
#Random sampling of association rules for comparison of confidence and lift values
rules_random=rules.sample(10, random_state = 42)
rules_lift = rules_random[['lift']].to_numpy()
rules_lift = (rules_lift/rules_lift.max()).transpose()[0]
rules_conf = rules_random[['confidence']].to_numpy()
rules_conf = (rules_conf/rules_conf.max()).transpose()[0]
width = 0.40
plt.figure(figsize=(12, 6), dpi=200)

# plot data in grouped manner of bar type
plt.bar(np.arange(len(rules_random))-0.2,rules_lift, width, color='black')
plt.bar(np.arange(len(rules_random))+0.2,rules_conf, width, hatch='//', edgecolor='black', facecolor='white')
plt.xlabel('Instance index')
plt.ylabel('Normalized metric value')
plt.legend(['lift','confidence'])
plt.xticks(range(0,10));