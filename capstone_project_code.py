import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


#Create your df here:
df = pd.read_csv('profiles.csv')

status_mapping = {'single':0,'available':1,'seeing someone':2,'married':3}
smokes_mapping = {'yes':0,'trying to quit':1,'sometimes':2,'when drinking':3,'no':4}
drinks_mapping = {'desperately':0,'very often':1,'often':2,'socially':3,'rarely':4,'not at all':5}
drugs_mapping = {'often':0,'sometimes':1,'never':2}
orientation_mapping = {'gay':0,'bisexual':1,'straight':2}
#body_mapping = {'overweight':0,"used up":0, 'full figured':1,'skinny':1,'a little extra':2,'thin':2, 'curvy':2,
#"average":3,"fit":4,'athletic':5,'jacked':6}

# Tried a different way of grouping the body types so that they will be in more 'equal' groups
# 0 = "below average", 1 = 'average', 2 = 'above average', 3 = 'ideal'
# This sort of grouping helps deal with human psychology, where people would rather not put thesmeves
# as a "bad" body_type. So everything that is 'worse than average' can be lumped together
body_mapping = {'overweight':0,"used up":0, 'full figured':0,'skinny':0,'a little extra':0,'thin':0, 'curvy':0,
"average":1,"fit":2,'athletic':3,'jacked':3}

df['status_code'] = df.status.map(status_mapping)
df['smokes_code'] = df.smokes.map(smokes_mapping)
df['drinks_code'] = df.drinks.map(drinks_mapping)
df['drugs_code'] = df.drugs.map(drugs_mapping)
df['orientation_code'] = df.orientation.map(orientation_mapping)
df['body_code'] = df.body_type.map(body_mapping)

df['income'] = df['income'].apply(lambda x: None if x ==-1 else x)

df['income_code'] = df['income'].apply(lambda x: 1 if x > 50000 else 0)

feature_data = df[['status_code','smokes_code','drinks_code','drugs_code','income_code']]

feature_data = pd.DataFrame.dropna(feature_data, axis=0, inplace=False)

labels = feature_data['income_code']
feature_data = feature_data.drop(['income_code'], axis=1)




# Plotting Visualizations

#plt.scatter(labels,feature_data['drugs_code'], alpha = 0.01)
#plt.xlabel('body_code')
#plt.ylabel('drugs')
#plt.title("Drugs vs Body Type")

''''
hm_data = df[['smokes_code', 'body_code']]
hm_data = pd.DataFrame.dropna(hm_data, axis=0, inplace=False)

print(hm_data.head(10))

hm_data = pd.crosstab(hm_data['smokes_code'], hm_data['body_code'])

data = np.array(hm_data.values)

print(data)

ax = sns.heatmap(data, linewidth=0.5)

plt.xlabel('body code')
plt.ylabel('smoking code')
plt.title("Smoking vs Body Type")

plt.show()
'''
x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)

# Splitting Test and Train Data
d_train, d_test, l_train, l_test = train_test_split(feature_data, labels, random_state=1)


# KNeighbors Classifier
accuracies = []
for i in range(1,1000,100):
    classifier = KNeighborsClassifier(i)
    classifier.fit(d_train, l_train)
    accuracies.append(classifier.score(d_test, l_test))
    print(accuracies)

k_list = range(1,1000,100)

plt.plot(k_list,accuracies)
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.title("Income >50k Classifier Accuracy")

plt.show()


# Random Forests
'''
forest = RandomForestClassifier(random_state = 1)
forest.fit(d_train, l_train)

print(forest.feature_importances_)

print(forest.score(d_test,l_test))
'''

# Linear Regression
'''
lm = LinearRegression()

model = lm.fit(d_train, l_train)

y_predict= lm.predict(d_test)

print("Train score:")
print(lm.score(d_train, l_train))

print("Test score:")
print(lm.score(d_test, l_test))

plt.scatter(l_test, y_predict)

plt.show()
'''