
cd_data = 'data/'
csv = 'titanictest.csv'
df = pd.read_csv(cd_data+csv)

df.select_dtypes

df = pd.DataFrame(df.drop(df.columns,axis =1)).merge(pd.get_dummies(df.drop(['Name','Cabin','Ticket'],axis =1)),left_index=True,right_index=True).drop(['PassengerId'],axis =1)
print(df.shape)
df.head()

print(df.isnull().sum())
plt.hist(df['Age']) # Identifying the distribution
np.random.seed(1111)
df['Age'] = df.fillna(pd.Series(np.random.chisquare(df['Age']))) # Filling in with bootstrapping

df.columns
X_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female',
       'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
X = df[X_columns]
y = df['Survived']

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
for learning_rate in learning_rates:
    gb = GradientBoostingClassifier(n_estimators=20, learning_rate = learning_rate, max_features=2, max_depth = 2, random_state = 0)
    gb.fit(X_train, y_train)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb.score(X_test, y_test)))
    print()

gb = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.1, max_features=2, max_depth = 2, random_state = 0)
gb.fit(X_train, y_train)

y_pred = gb.predict(X_test)


predicitons = pd.DataFrame(y_pred)
predicitons.columns = ['Survived']
predicitons.shape

predicitons.to_csv('output/pred.csv', index=False)
roc_curve(y_test, y_pred)

accuracy = str(round(sum(y_pred == y_test) / len(y_test), 6)*100)+'%'
print(accuracy)
plt.hist(y_pred, color='blue', alpha=0.5, bins=2, label='Predicition')
plt.hist(y_test, color='red', alpha=0.5, bins=2, label='Test')
plt.xticks([0,1])
plt.title('Predicition VS Test')
plt.xlabel('Survived')
plt.ylabel('Value Counts')
plt.text(-0.09, 150, 'Accuracy: '+str(accuracy))
plt.legend()
plt.savefig('figures/HistTitanicPredVSTest_real.png')
