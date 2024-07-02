# %% importovi

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
#from datetime import datetime
from dateutil.parser import parse
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
def date_parser(x): return parse(x)

warnings.filterwarnings("ignore")
# %% HOTENCODE FOO


def hotencode(df, colname):
    save = df[colname]
    n = np.unique(df[colname])
    idx = 1
    for i in n:
        df[colname] = df[colname].apply(lambda a: idx if a == i else a)
        idx += 1
    return save


# %% ucitavanje podataka


path = Path(__file__).parent / "Crime.csv"

df = pd.read_csv(path)


print("Shape dataseta pre uklanjanja null: {}", df.shape)

if (pd.isna(df).values.any()):
    df = df.dropna()
    print("Uklonjene null vrednosti")

droppable = ['Occurred Through Date']
df = df.drop(droppable, axis=1)

print("Shape dataseta posle uklanjanja null i end date: {}", df.shape)


df = df.rename({'Occurred From Date': 'Ocurred Date'})

print(df.dtypes)

print(df["Case Number"])

df['Case Number'] = df['Case Number'].apply(lambda x: x.replace('-', ''))
df['Case Number'] = pd.to_numeric(df['Case Number'], errors='coerce')

print(df["Case Number"])

# %% parsiranje podataka

# Sredjivanje Occurred Time

df['Occurred Time'] = df['Occurred From Date'].str.split(' ').str[1]
df['Occurred Date'] = df['Occurred From Date'].str.split(' ').str[0]
df['Date No Year'] = df['Occurred Date'].str.split('-').str[2] + '-' + df['Occurred Date'].str.split('-').str[1] 
df['Occurred Day'] = df['Occurred Date'].str.split('-').str[2].astype(int)
df['Occurred Month'] = df['Occurred Date'].str.split('-').str[1].astype(int)
df['Occurred Year'] = df['Occurred Date'].str.split('-').str[0].astype(int)
df['Occurred Date'] = df['Occurred Date'].apply(date_parser)


# Sredjivanje Crime Type

# Split
#df['Crime Subtype'] = df['Crime Type'].str.split('-').str[1].str.strip().astype(str)
df['Crime Type'] = df['Crime Type'].str.split('-').str[0].str.strip().astype(str)

# Group
df['Crime Type'] = df['Crime Type'].replace(
    'ASSAULTS (PRIOR TO SEPT 2018)', 'ASSAULT')
df['Crime Type'] = df['Crime Type'].replace('SHOPLIFT ROBBERY', 'SHOPLIFTING')
df['Crime Type'] = df['Crime Type'].replace(
    'THEFT FROM COMMERCIAL BUILDING', 'THEFT')
df['Crime Type'] = df['Crime Type'].replace('THEFT OF AUTO PARTS', 'THEFT')
df['Crime Type'] = df['Crime Type'].replace('MOTOR VEHICLE THEFT', 'THEFT')


# Parsiranje Preostalih kolona

df['Case Number'] = df['Case Number'].astype(str)
df['Block Address'] = df['Block Address'].astype(str)


print("Oblik dataseta posle razdvajanja kolona:", df.shape)

#%% SKIP
# Sredjivanje Reported Date

#print(df['Reported Date'])
#df['Reported Time'] = ((df["Reported Date"].str.split(' ').str[3]))
#df['Reported Time'] = df['Reported Time'].apply(lambda a: datetime.strptime(a, '%H:%M:%S').time())
#df['Reported Date'] = df["Reported Date"].str.split(' ').str[0].apply(date_parser)


#%% APRIORI ZA ANALIZIRANJE 
#%%
# Najcesci zlocini na osnovu lokacije

basket = df.groupby(['Block Address', 'Crime Type']).size().unstack(fill_value=0)
basket = basket.applymap(lambda x: x > 0)  
print (basket)

min_support_value = 0.01  
frequent_itemsets = apriori(basket, min_support=min_support_value, use_colnames=True)
print(frequent_itemsets)

metric_type = "lift"  
min_threshold_value = 1
rules = association_rules(frequent_itemsets, metric=metric_type, min_threshold=min_threshold_value)

print (rules)
# Ispisivanje rezultata
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)
print(frequent_itemsets.sort_values(by='support', ascending=False).head(10))

# Prikaz najcescih
frequent_itemsets['itemsets_str'] = frequent_itemsets['itemsets'].apply(lambda x: set(x))

plt.figure(figsize=(10, 6))
plt.bar(frequent_itemsets['itemsets_str'][:10].astype(str), frequent_itemsets['support'][:10],  facecolor='black')
plt.xlabel('Itemsets')
plt.ylabel('Support')
plt.title('Most frequent types of crime/crime combination to occur per location')
plt.xticks(rotation=45, ha='right')
plt.axhline(y=0.3, color='red', linewidth=3, linestyle='dotted', label='Threshold')
plt.tight_layout()
plt.show()


#Prikaz nasumicnih rezultata radi interpretacije
rules_random=rules.sample(10, random_state = 42)
rules_lift = rules_random[['lift']].to_numpy()
rules_acters = rules_random[['antecedents', 'consequents']]
rules_lift = (rules_lift/rules_lift.max()).transpose()[0]
width = 0.40
plt.figure(figsize=(12, 6), dpi=200)

plt.bar(np.arange(len(rules_random)),rules_lift, width, color='black')
plt.axhline(y=0.5, color='red', linewidth=3, linestyle='dotted', label='Threshold')
plt.xlabel('Instance index')
plt.ylabel('Lift values when analyzing crime types based on location')

plt.xticks(range(0,10));

rules_data = {
    'Antecedents': [', '.join(map(str, rule)) for rule in rules_acters['antecedents']],
    'Consequents': [', '.join(map(str, rule)) for rule in rules_acters['consequents']]
}

rules_table = pd.DataFrame(rules_data)
print(rules_table)


#%% Analiza dela godine
basket = df.groupby(['Block Address', 'Occurred Month']).size().unstack(fill_value=0)
basket = basket.applymap(lambda x: x > 0)

min_support_value = 0.01  
frequent_itemsets = apriori(basket, min_support=min_support_value, use_colnames=True)

metric_type = "lift"  
min_threshold_value = 1
rules = association_rules(frequent_itemsets, metric=metric_type, min_threshold=min_threshold_value)

print (rules)
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)
print(frequent_itemsets.sort_values(by='support', ascending=False).head(10))

frequent_itemsets['itemsets_str'] = frequent_itemsets['itemsets'].apply(lambda x: set(x))

plt.figure(figsize=(10, 6))
plt.bar(frequent_itemsets['itemsets_str'][:15].astype(str), frequent_itemsets['support'][:15],  facecolor='black')
plt.xlabel('Itemsets')
plt.ylabel('Support')
plt.title('Most frequent time of the year per location')
plt.xticks(rotation=45, ha='right')
plt.axhline(y=0.3, color='red', linewidth=3, linestyle='dotted', label='Threshold')
plt.tight_layout()
plt.show()



#Prikaz nasumicnih rezultata radi interpretacije
rules_random=rules.sample(10, random_state = 42)
rules_lift = rules_random[['lift']].to_numpy()
rules_acters = rules_random[['antecedents', 'consequents']]
rules_lift = (rules_lift/rules_lift.max()).transpose()[0]
width = 0.40
plt.figure(figsize=(12, 6), dpi=200)

plt.bar(np.arange(len(rules_random)),rules_lift, width, color='black')
plt.axhline(y=0.5, color='red', linewidth=3, linestyle='dotted', label='Threshold')
plt.xlabel('Instance index')
plt.ylabel('Normalized lift value')
plt.title("Lift values when analyzing most frequent time of the year when crimes occurred")
plt.xticks(range(0,10));

rules_data = {
    'Antecedents': [', '.join(map(str, rule)) for rule in rules_acters['antecedents']],
    'Consequents': [', '.join(map(str, rule)) for rule in rules_acters['consequents']]
}

rules_table = pd.DataFrame(rules_data)

print(rules_table)

#%% Najcesci zlocini i njihove kombinacije generalno, u vremenu
basket = df.groupby(['Occurred Date', 'Crime Type']).size().unstack(fill_value=0)
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
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)
print(frequent_itemsets.sort_values(by='support', ascending=False).head(10))

frequent_itemsets['itemsets_str'] = frequent_itemsets['itemsets'].apply(lambda x: set(x))

plt.figure(figsize=(10, 6))
plt.bar(frequent_itemsets['itemsets_str'][:15].astype(str), frequent_itemsets['support'][:15],  facecolor='black')
plt.xlabel('Itemsets')
plt.ylabel('Support')
plt.title('Most frequent crimes')
plt.xticks(rotation=45, ha='right')
plt.axhline(y=0.3, color='red', linewidth=3, linestyle='dotted', label='Threshold')
plt.tight_layout()
plt.show()


#Prikaz nasumicnih rezultata radi interpretacije
rules_random=rules.sample(10, random_state = 42)
rules_lift = rules_random[['lift']].to_numpy()
rules_acters = rules_random[['antecedents', 'consequents']]
rules_lift = (rules_lift/rules_lift.max()).transpose()[0]
width = 0.40
plt.figure(figsize=(12, 6), dpi=200)

plt.bar(np.arange(len(rules_random)),rules_lift, width, color='black')
plt.axhline(y=0.5, color='red', linewidth=3, linestyle='dotted', label='Threshold')
plt.xlabel('Instance index')
plt.ylabel('Normalized lift value')
plt.title("Lift values when analyzing most frequent crimes and their relations")
plt.xticks(range(0,10));

rules_data = {
    'Antecedents': [', '.join(map(str, rule)) for rule in rules_acters['antecedents']],
    'Consequents': [', '.join(map(str, rule)) for rule in rules_acters['consequents']]
}

rules_table = pd.DataFrame(rules_data)

print()
print(rules_table)

#%% Analiziranje najcesceg dela dana kada se zlocini desavaju

#Sredjivanje nove kolone 
print (np.unique(df['Occurred Time']))
df['hour'] = df['Occurred Time'].str.split(':').str[0].astype(int)
print (df['hour'])
df['Time Of The Day'] = df['hour'].apply(lambda x: 'Late Night' if x <= 4 else 'Early Morning' if x <= 8 else 'Morning' if x <= 12 else 'Afternoon' if x<=16 else 'Evening' if x <= 20 else 'Night')

basket = df.groupby(['Block Address', 'Time Of The Day']).size().unstack(fill_value=0)
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
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)
print(frequent_itemsets.sort_values(by='support', ascending=False).head(10))

frequent_itemsets['itemsets_str'] = frequent_itemsets['itemsets'].apply(lambda x: set(x))

plt.figure(figsize=(10, 6))
plt.bar(frequent_itemsets['itemsets_str'][:15].astype(str), frequent_itemsets['support'][:15],  facecolor='black')
plt.xlabel('Itemsets')
plt.ylabel('Support')
plt.title('Most frequent time of the crimes')
plt.xticks(rotation=45, ha='right')
plt.axhline(y=0.3, color='red', linewidth=3, linestyle='dotted', label='Threshold')
plt.tight_layout()
plt.show()

#Prikaz nasumicnih rezultata radi interpretacije
rules_random=rules.sample(10, random_state = 42)
rules_lift = rules_random[['lift']].to_numpy()
rules_acters = rules_random[['antecedents', 'consequents']]
rules_lift = (rules_lift/rules_lift.max()).transpose()[0]
width = 0.40
plt.figure(figsize=(12, 6), dpi=200)

plt.bar(np.arange(len(rules_random)),rules_lift, width, color='black')
plt.axhline(y=0.5, color='red', linewidth=3, linestyle='dotted', label='Threshold')
plt.xlabel('Instance index')
plt.ylabel('Normalized lift value')
plt.title("Lift values when analyzing most frequent time of crime")
plt.xticks(range(0,10));

rules_data = {
    'Antecedents': [', '.join(map(str, rule)) for rule in rules_acters['antecedents']],
    'Consequents': [', '.join(map(str, rule)) for rule in rules_acters['consequents']]
}

rules_table = pd.DataFrame(rules_data)
print()
print(rules_table)

#%% ARIMA I PREDVIDJANJE
#%% mapiranje stringova na intove

og_address = hotencode(df, 'Block Address')
og_type = hotencode(df, "Crime Type")


print(df.dtypes)

# %% Izdvajanje kolona za laksi rad

x = df[['Case Number','Block Address', 'Occurred Date', 'Crime Type', 'Occurred Year', 'Date No Year']]

# %% 2024 CNT

count_2224 = x['Occurred Year'].value_counts().get(
    2022) + x['Occurred Year'].value_counts().get(2023) + x['Occurred Year'].value_counts().get(2024)

count_2224_percent = (count_2224 / df.shape[0])*100

print(
    f'The number of occurrences in the year 2024 is: {count_2224}; this is {round(count_2224_percent,2)}% of data')

print(
    f'This makes Test:Train ratio approximately {round(100-count_2224_percent)}:{round(count_2224_percent)}')


# %% Razdvajanje na test i train skup
print("\n\nTraining skup su podaci do 2022 sto je ~76%, refer to previous cell for details!")
train = x[x['Occurred Year'] < 2022]
test = x[x['Occurred Year'] >= 2022]
print("Train and test shapes after split: \n\tTest",
      test.shape, "\n\tTrain", train.shape)



# %% Predvidjanje broja zlocina na odnosu lokacije aka predvidjanje najproblematicnijeg dela grada

train_df = train.groupby(['Block Address', 'Crime Type']).size().unstack(fill_value=0)
train_df['Total'] = train_df.iloc[:, 2:].sum(axis=1)

test_df = test.groupby(['Block Address', 'Crime Type']).size().unstack(fill_value=0)
test_df['Total'] = test_df.iloc[:, 2:].sum(axis=1)

plot_pacf(train_df['Total'], lags=10, method='ols')
plt.show()

plot_acf(train_df['Total'], lags=10)
plt.show()



p, d, q = 11, 3, 6
arima_model = ARIMA(train_df['Total'], order=(p, d, q)).fit()

y_pred_arima = arima_model.predict(start=test_df.index[0], end=test_df.index[-1])

y_pred_arima = y_pred_arima * 1


#plt.plot(train_df['Total'], color='b', linewidth=4, alpha=0.3, label='train')
plt.plot(test_df['Total'], color='mediumblue',
         linewidth=4, alpha=0.3, label='test')
plt.plot(y_pred_arima, color='b', label='ARIMA model prediction')
plt.title('predikcije za Total')
plt.legend()
plt.xlim(0,1000)
plt.show()

#%% PREDVIDJANJE NAJPROBLEM DOBA GODINE 

train_df = train.groupby(['Date No Year', 'Crime Type']).size().unstack(fill_value=0)
train_df['Total'] = train_df.iloc[:, 2:].sum(axis=1)

test_df = test.groupby(['Date No Year', 'Crime Type']).size().unstack(fill_value=0)
test_df['Total'] = test_df.iloc[:, 2:].sum(axis=1)

#train_df = np.log10(train_df[['Total']])


plot_pacf(train_df['Total'], lags=35, method='ols')
plt.show()

plot_acf(train_df['Total'], lags=35)
plt.show()

p, d, q = 10,4, 11
arima_model = ARIMA(train_df['Total'], order=(p, d, q)).fit()

y_pred_arima = arima_model.predict(
    start=test_df.index[0], end=test_df.index[-1])
#y_pred_arima = np.power(y_pred_arima,10)

#plt.plot(train_df['Total'], color='b', linewidth=4, alpha=0.3, label='train')
plt.plot(test_df['Total'][20:], color='mediumblue',
         linewidth=1, alpha=0.3, label='test')
plt.plot(y_pred_arima[20:], color='b', label='ARIMA model')
plt.title('ARIMA prediction for the crime occurrance date (d = 4)')
plt.legend()
plt.xlim(0,300)
plt.xticks(ticks = range(1,299))
plt.show()


