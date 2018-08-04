## Easy Grid Search / Cross Validation

This library allows you to quickly train machine learning classifiers by 
automatically splitting the dataset and using both
[grid search](https://en.wikipedia.org/wiki/Hyperparameter_optimization) and [cross validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) in the training process. 

Users can either pass define the parameters themselves or let the **GSCV** object
choose them automatically (based on the classifier). 

This library is an extension of the [scikit-learn](http://scikit-learn.org/stable/index.html) project. 


### Example:
```
params = {}
model = GSCV(GradientBoostingClassifier(), X, y, cv=10, n_jobs=1, params=None)

```

#### create
```
```

## score
```
```

## get_best_estimator
```
```

## get_fit_details
```
```
