[![Build Status](https://travis-ci.org/Jasper-Koops/easy-gscv.svg?branch=master)](https://travis-ci.org/Jasper-Koops/easy-gscv)
[![codecov](https://codecov.io/gh/Jasper-Koops/easy-gscv/branch/master/graph/badge.svg)](https://codecov.io/gh/Jasper-Koops/easy-gscv)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

## Easy Grid Search / Cross Validation

*From data to score in 4 lines of code.*

This library allows you to quickly train machine learning classifiers by
automatically splitting the dataset and using both
[grid search](https://en.wikipedia.org/wiki/Hyperparameter_optimization) and [cross validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) in the training process. Users can either pass define the parameters themselves or let the **GSCV** object
choose them automatically (based on the classifier).

This library is an extension of the [scikit-learn](http://scikit-learn.org/stable/index.html) project.

[View on pypi](https://pypi.org/project/easy-gscv/)


### Example:

```
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from easy_gscv.classifiers import GSCV

# Create test dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
clf = MLPClassifier()

# Create model instance
gscv_model = GSCV(clf(), X, y)

# Get score
gscv_model.score()

```


## install

*requires python 3.7+*

```
pip install easy-gscv
```


## create

```
from easy_gscv.models import GSCV
clf = LogisticRegression()
gscv_model = GSCV(
    clf(), X, y, cv=15, n_jobs=-1, params={
        'C': [10, 100],
        'penalty': ['l2']
    }
)
```

No need to create separate train / test datasets, the model does this
automatically on initialization.
If no parameters are provided the grid search is performed on a default set.
But these can be overridden.

The number of folds to be used for cross validation can be specified
by using the `cv` keyword.
To speed up the training process you can use the `n_jobs` parameter to
set the number of cpu cores to use (or set it to `-1` to use all available.)

The model accepts either sklearn classifiers or string values.
You can get a list of valid classifiers by calling the 'classifiers' property. Passing string arguments to the GSCV object in turn saves
you from having to import sklearn classifiers yourself.

```
gscv_model = GSCV('RandomForestClassifier',, X, y)
gscv_model.classifiers

'KNeighborsClassifier',
'RandomForestClassifier',
'GradientBoostingClassifier',
'MLPClassifier',
'LogisticRegression',
```


## score

```
gscv_model.score()
```

The grid search is performed on the training data. Use the `score` method to evaluate
how well the model can be generalized by scoring it against the test dataset.


## get_best_estimator

```
gscv_model.get_best_estimator()
```

Returns the best scoring sklearn classifier (based on training data).
As its a valid scikit-learn classifier, you can use it do anything that
you could do with sklearn classifier.

The following classifiers are currently supported. With the eventual goal of
supporting all scikit-learn classifiers in the future.

* KNeighborsClassifier
* RandomForestClassifier
* GradientBoostingClassifier
* MLPClassifier
* LogisticRegression


## get_fit_details

As cross validation returns an average, it can be helpful to
get a more detailed overview of the best scoring classifier.

This method returns a table like the one displayed below, which
then can be used to further refine the choice or parameters for
subsequent runs.

```
clf = KNeighborsClassifier()
gscv_model = GSCV(clf(), X, y)
gscv_model.get_fit_details()

0.965 (+/-0.026) for {'weights': 'uniform', 'n_neighbors': 3}
0.977 (+/-0.013) for {'weights': 'distance', 'n_neighbors': 3}
0.979 (+/-0.011) for {'weights': 'uniform', 'n_neighbors': 5}
0.979 (+/-0.011) for {'weights': 'distance', 'n_neighbors': 5}
0.976 (+/-0.018) for {'weights': 'uniform', 'n_neighbors': 8}
0.975 (+/-0.018) for {'weights': 'distance', 'n_neighbors': 8}
0.971 (+/-0.022) for {'weights': 'uniform', 'n_neighbors': 12}
0.973 (+/-0.024) for {'weights': 'distance', 'n_neighbors': 12}
0.973 (+/-0.025) for {'weights': 'uniform', 'n_neighbors': 15}

```
