Here is a list of common packages to use during a data science project. Packages are listed depending on the task to accomplish.

# Visualization
* Scatter matrix `from pandas.plotting import scatter_matrix`

# Preprocessing
* Pipelines `from sklearn.pipeline import Pipeline`
* Standardization `from sklearn.preprocessing import StandardScaler`


# Models
## Linear algorithms
* ElasticNet `from sklearn.linear_model import ElasticNet`
* Lasso `from sklearn.linear_model import Lasso`
* LDA `from sklearn.discriminant_analysis import LinearDiscriminantAnalysis`
* Linear regression `from sklearn.linear_model import LinearRegression`
* Logistic regression `from sklearn.linear_model import LogisticRegression`

## Non linear algorithms
* CART `from sklearn.tree import DecisionTreeClassifier`
* Gradient Boosted Trees `from sklearn.ensemble import GradientBoostingClassifier`
* KNN `from sklearn.neighbors import KNeighborsClassifier`
* Naive Bayes `from sklearn.naive_bayes import GaussianNB`
* Random Forest `from sklearn.ensemble import RandomForestClassifier`
* SVM `from sklearn.svm import SVC`


# Model evaluation (split data, CV, scoring)
## Cross-validation
* Grid search `from sklearn.model_selection import GridSearchCV`
* Perform CV `from sklearn.model_selection import cross_val_score` [doc_here](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)
* Split Train / test sets `from sklearn.model_selection import train_test_split` [doc here](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
* Timeseries split `from sklearn.model_selection import TimeSeriesSplit`

## Scoring / results
* Accuracy score `from sklearn.metrics import accuracy_score`
* Confusion matrix `from sklearn.metrics import confusion_matrix`
* Classification report `from sklearn.metrics import classification_report`
