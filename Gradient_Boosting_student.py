# %% codecell
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn import tree
from IPython.display import Image
%matplotlib inline
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
# %% markdown
# ### Gradient boosting
#
# You may recall that we last encountered gradients when discussing the gradient descent algorithm in the context of fitting linear regression models.  For a particular regression model with n parameters, an n+1 dimensional space existed defined by all the parameters plus the cost/loss function to minimize.  The combination of parameters and loss function define a surface within the space.  The regression model is fitted by moving down the steepest 'downhill' gradient until we reach the lowest point of the surface, where all possible gradients are 'uphill.'  The final model is made up of the parameter estimates that define that location on the surface.
#
# Throughout all iterations of the gradient descent algorithm for linear regression, one thing remains constant: The underlying data used to estimate the parameters and calculate the loss function never changes.  In gradient boosting, however, the underlying data do change.
#
# Each time we run a decision tree, we extract the residuals.  Then we run a new decision tree, using those residuals as the outcome to be predicted.  After reaching a stopping point, we add together the predicted values from all of the decision trees to create the final gradient boosted prediction.
#
# Gradient boosting can work on any combination of loss function and model type, as long as we can calculate the derivatives of the loss function with respect to the model parameters.  Most often, however, gradient boosting uses decision trees, and minimizes either the  residual (regression trees) or the negative log-likelihood (classification trees).
#
# Let’s go through a simple regression example using Decision Trees as the base predictors (of course Gradient Boosting also works great with regression tasks). This is called Gradient Tree Boosting, or Gradient Boosted Regression Trees. First, let’s fit a `DecisionTreeRegressor` to the training set.
# %% codecell
np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3*X[:, 0]**2 + 0.05 * np.random.randn(100)
# %% codecell
from sklearn.tree import DecisionTreeRegressor

tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)
# %% markdown
# Now train a second `DecisionTreeRegressor` on the residual errors made by the first predictor:
# %% codecell
y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg2.fit(X, y2)
# %% markdown
# Then we train a third regressor on the residual errors made by the second predictor:
#
#
# %% codecell
y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg3.fit(X, y3)
# %% markdown
# Now we have an ensemble containing three trees. It can make predictions on a new instance simply by adding up the predictions of all the trees:
# %% codecell
X_new = np.array([[0.8]])
# %% codecell
y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
# %% codecell
y_pred
# %% markdown
# The figure below represents the predictions of these three trees in the left column, and the ensemble’s predictions in the right column. In the first row, the ensemble has just one tree, so its predictions are exactly the same as the first tree’s predictions. In the second row, a new tree is trained on the residual errors of the first tree. On the right you can see that the ensemble’s predictions are equal to the sum of the predictions of the first two trees. Similarly, in the third row another tree is trained on the residual errors of the second tree. You can see that the ensemble’s predictions gradually get better as trees are added to the ensemble.
# %% markdown
# **<font color='teal'>Run the below cell to develop a visual representation.</font>**
# %% codecell
def plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc="upper center", fontsize=16)
    plt.axis(axes)

plt.figure(figsize=(11,11))

plt.subplot(321)
plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h_1(x_1)$", style="g-", data_label="Training set")
plt.ylabel("$y$", fontsize=16, rotation=0)
plt.title("Residuals and tree predictions", fontsize=16)

plt.subplot(322)
plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1)$", data_label="Training set")
plt.ylabel("$y$", fontsize=16, rotation=0)
plt.title("Ensemble predictions", fontsize=16)

plt.subplot(323)
plot_predictions([tree_reg2], X, y2, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_2(x_1)$", style="g-", data_style="k+", data_label="Residuals")
plt.ylabel("$y - h_1(x_1)$", fontsize=16)

plt.subplot(324)
plot_predictions([tree_reg1, tree_reg2], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1)$")
plt.ylabel("$y$", fontsize=16, rotation=0)

plt.subplot(325)
plot_predictions([tree_reg3], X, y3, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_3(x_1)$", style="g-", data_style="k+")
plt.ylabel("$y - h_1(x_1) - h_2(x_1)$", fontsize=16)
plt.xlabel("$x_1$", fontsize=16)

plt.subplot(326)
plot_predictions([tree_reg1, tree_reg2, tree_reg3], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1) + h_3(x_1)$")
plt.xlabel("$x_1$", fontsize=16)
plt.ylabel("$y$", fontsize=16, rotation=0)

#save_fig("gradient_boosting_plot")
plt.show()
# %% markdown
# Now that you have solid understanding of Gradient Boosting in the regression scenario, let's apply the same algorithm to a classification problem. Specifically, the Titanic dataset and predicting survival.
# %% markdown
# **<font color='teal'>Use pandas read csv to load in the Titantic data set into a dataframe called df.</font>**
# %% codecell
cd_data = 'data/'
train = 'titanictrain.csv'
test = 'titanictest.csv'
df_train = pd.read_csv(cd_data+train)
df_test = pd.read_csv(cd_data+test)
df = pd.concat([df_train, df_test])
df.head()
# %% markdown
# **<font color='teal'>Print the levels of the categorical data using 'select_dtypes'. </font>**
# %% codecell
df.select_dtypes
# %% markdown
# **<font color='teal'>Create dummy features for the categorical features and add those to the 'df' dataframe. Make sure to also remove the original categorical columns from the dataframe.</font>**
# %% codecell
df = pd.DataFrame(df.drop(df.columns,axis =1)).merge(pd.get_dummies(df.drop(['Name','Cabin','Ticket'],axis =1)),left_index=True,right_index=True).drop(['PassengerId'],axis =1)
print(df.shape)
df.head()
# %% markdown
# **<font color='teal'>Print the null values for each column in the dataframe.</font>**
# %% codecell
print(df.isnull().sum())


plt.hist(df['Age']) # Identifying the distribution
np.random.seed(1111)
df['Age'] = df.fillna(np.mean(df['Age']))
df['Fare'] = df.fillna(np.mean(df['Fare']))
df['Survived'] = df.fillna(float(np.random.randint(1)))

# %% markdown
# **<font color='teal'>Create the X and y matrices from the dataframe, where y = df.Survived </font>**
# %% codecell
df.columns
X_columns = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female',
       'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
X = df[X_columns]
y = df['Survived']
# X = df_train
# y = df_test
# %% markdown
# **<font color='teal'>Apply the standard scaler to the X matrix.</font>**
# %% codecell
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
# %% markdown
# **<font color='teal'>Split the X_scaled and y into 75/25 training and testing data subsets..</font>**
# %% codecell
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# %% markdown
# **<font color='teal'>Run the cell below to test multiple learning rates in your gradient boosting classifier.</font>**
# %% codecell
learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
for learning_rate in learning_rates:
    gb = GradientBoostingClassifier(n_estimators=20, learning_rate = learning_rate, max_features=2, max_depth = 2, random_state = 0)
    gb.fit(X, y)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb.score(X, y)))
    print("Accuracy score (validation): {0:.3f}".format(gb.score(X, y)))
    print()
# %% markdown
# **<font color='teal'>Apply the best learning rate to the model fit and make some predictions. If you like, you can also calculate the ROC for your model. To evaluate your model, submit it to the (now very famous) [Kaggle competition](https://www.kaggle.com/c/titanic/) that Professor Spiegelhalter references in Chapter 5 of The Art of Statistics.</font>**
# %% codecell
gb = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.056, max_features=2, max_depth = 2, random_state = 0)
gb.fit(X, y)

y_pred = gb.predict(X)


predicitons = pd.DataFrame(y_pred)
predicitons.index.rename('PassengerId', inplace=True)
predicitons.reset_index(inplace=True)
predicitons.columns = ['Survived']
predicitons.shape

predicitons.to_csv('output/pred.csv', index=False)
roc_curve(y, y_pred)

accuracy = str(round(sum(y_pred == y) / len(y), 6)*100)+'%'
print(accuracy)
plt.hist(y_pred, color='blue', alpha=0.5, bins=2, label='Predicition')
plt.hist(y, color='red', alpha=0.5, bins=2, label='Test')
plt.xticks([0,1])
plt.title('Predicition VS Test')
plt.xlabel('Survived')
plt.ylabel('Value Counts')
plt.text(-0.09, 150, 'Accuracy: '+str(accuracy))
plt.legend()
plt.savefig('figures/HistTitanicPredVSTest.png')

print(len(predicitons))
