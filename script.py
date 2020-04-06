import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
df = pd.read_csv('tennis_stats.csv')
print(df.head())

# perform exploratory analysis here:

# plt.scatter(df['BreakPointsOpportunities'], df['Winnings'])
# plt.scatter = (df['ServiceGamesPlayed'], df['Wins'])
# plt.show()

# perform single feature linear regressions here:
X = df['ServiceGamesPlayed']
X=X.values.reshape(-1,1)
y = df['Winnings']
plt.scatter(X,y)

X_train, X_test, y_train, y_test = train_test_split(X,y)
model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_train, y_train))

y_pred = model.predict(X_train)
print(model.score(X_test, y_test))


X1 = df['ReturnGamesPlayed']
X1=X1.values.reshape(-1,1)
y1 = df['Winnings']
plt.scatter(X1,y1)

X1_train, X1_test, y1_train, y1_test = train_test_split(X1,y1)
model1 = LinearRegression()
model1.fit(X1_train, y1_train)
print(model1.score(X1_train, y1_train))

y1_pred = model1.predict(X1_train)
print(model1.score(X1_test, y1_test))

X2 = df['DoubleFaults']
X2=X2.values.reshape(-1,1)
y2 = df['Winnings']
plt.scatter(X2,y2)

X2_train, X2_test, y2_train, y2_test = train_test_split(X2,y2)
model2 = LinearRegression()
model2.fit(X2_train, y2_train)
print(model2.score(X2_train, y2_train))

y2_pred = model2.predict(X2_train)
print(model2.score(X2_test, y2_test))

# perform two feature linear regressions here:
features = df[['ServiceGamesPlayed', 'DoubleFaults']]
outcome = df[['Winnings']]

features_train, features_test, outcome_train, outcome_test = train_test_split(features,outcome)
tmodel = LinearRegression()
tmodel.fit(features_train, outcome_train)
print(tmodel.score(features_train, outcome_train))

outcome_pred = tmodel.predict(features_test)
print(tmodel.score(features_test, outcome_test))



# perform multiple feature linear regressions here:
mfeatures = df[['ServiceGamesPlayed', 'DoubleFaults', 'ReturnGamesPlayed', 'BreakPointsOpportunities']]
moutcome = df[['Winnings']]

mfeatures_train, mfeatures_test, moutcome_train, moutcome_test = train_test_split(mfeatures,moutcome)
mmodel = LinearRegression()
mmodel.fit(mfeatures_train, moutcome_train)
print("My result in multiple regression for training dataset:",mmodel.score(mfeatures_train, moutcome_train))

moutcome_pred = mmodel.predict(mfeatures_test)
print("My result in multiple regression for validation dataset:", mmodel.score(mfeatures_test, moutcome_test))




















