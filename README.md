## Easy Grid Search / Cross Validation

*From data to score in 4 lines of code.*

This library allows you to quickly train machine learning classifiers by 
automatically splitting the dataset and using both
[grid search](https://en.wikipedia.org/wiki/Hyperparameter_optimization) and [cross validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) in the training process. 

Users can either pass define the parameters themselves or let the **GSCV** object
choose them automatically (based on the classifier). 

This library is an extension of the [scikit-learn](http://scikit-learn.org/stable/index.html) project. 


### Example:
```
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from easy_gscv.models import GSCV

# Create test dataset
iris = datasets.load_iris()
self.X = iris.data
self.y = iris.target
clf = MLPClassifier()

# Create model instance
gscv_model = GSCV(clf(), X, y, cv=10, n_jobs=-1)

# Get score 
gscv_model.score()

neural_net_classifier = gscv_model.get_best_estimator()
```

#### create

No need to create separate train / test datasets, the model does this
automatically. 
```
```

## score

```
gscv_model.score()
```

## get_best_estimator

Returns a trained sklearn classifier
```
gscv_model.get_best_estimator()
```

## get_fit_details
```
gscv_model.get_fit_details()
```
