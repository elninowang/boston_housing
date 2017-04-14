# _*_ coding:utf-8 _*_
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, make_scorer, accuracy_score, explained_variance_score
from sklearn.tree import DecisionTreeRegressor

# 加载北京数据
data = pd.read_csv('bj_housing.csv')
prices = data['Value'].as_matrix()
features = data.drop('Value', axis=1).as_matrix()

print "Beijing housing dataset has {} data points with {} variables each.".format(*data.shape)

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between
        true and predicted values based on the metric chosen. """
    score = r2_score(y_true, y_predict)
    return score

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a
        decision tree regressor trained on the input data [X, y]. """
    cv_sets = ShuffleSplit(n_splits=10, test_size=0.20, random_state=42)

    regressor = DecisionTreeRegressor()
    params = {'max_depth':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    scoring_fnc = make_scorer(performance_metric)
    grid = GridSearchCV(regressor, params, scoring=scoring_fnc, cv=cv_sets)
    grid = grid.fit(X, y)
    return grid.best_estimator_, grid.best_score_

def fit_model_with_max_depth(X, y, max_depth):
    regressor = DecisionTreeRegressor(max_depth=max_depth)
    scoring_fnc = make_scorer(performance_metric)
    regressor.fit(X, y)
    return regressor

X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2)

for i in range(1,11):
    ### max-depth=i 的方案
    reg = fit_model_with_max_depth(X_train, y_train, i)
    y_pred_max_depth = reg.predict(X_test).astype(int)
    print "Max_depth={} test r2_score is {}".format(i, r2_score(y_test, y_pred_max_depth))
print ""

### 采用GridSearchCV 来做
reg_best, score = fit_model(X_train, y_train)
# 打印出GridSearchCV 自动匹配出来的深度参数
print("GridSearchCV Parameter 'max_depth' is {} for the optimal model. and fix r2_score is {}".format(reg_best.get_params()['max_depth'], score))
y_pred = reg_best.predict(X_test).astype(int)
print "GridSearchCV test r2_score is ", r2_score(y_test, y_pred)