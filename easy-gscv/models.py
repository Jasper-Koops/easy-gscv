from sklearn.model_selection import train_test_split, GridSearchCV
from typing import Optional, Dict, Union


class GSCV:

    def __init__(
        self, clf, X, y, cv: int=10, random_state: int=42,
        test_size: float=0.33, n_jobs: int=1,
        params: Optional[Dict[str, Union[str, float]]]=None
    ):
        # Split data into Train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=random_state
        )

        if params is None:
            params = self._get_model_params()

        self.clf = clf
        self.params = params
        self.X_train = X_train
        self.y_train = y_train
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _get_model_params(self):
        """
        An Internal method to get a matching model_param dict for the
        provided classifier, called only if no params are provided.
        """
        # Get number of features
        if self.X_train.shape[1] > 10:
            no_ftrs = self.X_train.shape[1]
        else:
            no_ftrs = 10

        # Choice of algorithm depends on size of dataset
        if self.X_train.shape[1] < 500:
            solver = 'lbfgs' 
        else:
            solver = 'adam'

        param_dict = {
            'KNeighborsClassifier': {
                'n_neighbors': [3, 5, 8, 10, 15],
                'weights': ['uniform', 'distance'],
            },
            'RandomForestClassifier': {
                'n_estimators': [100, 500, 1000],
                'max_features': ['sqrt', 'log2', None],
                'max_depth': [None, 3, 5],
            },
            'GradientBoostingClassifier': {
                'n_estimators': [100, 500, 1000],
                'learning_rate': [0.1, 0.5, 1],
                'max_depth': [1, 3, 5],
                'max_features': ['sqrt', 'log2', None],
            },
            'MLPClassifier': {
                'hidden_layer_sizes': [
                    (no_ftrs,),
                    (no_ftrs, no_ftrs),
                    (no_ftrs, no_ftrs, no_ftrs)
                ],
                'alpha': [0.0001, 0.01, 0.1, 1],
                'solver': [solver],
            },
            'LogisticRegression': {
                'C': [0.01, 1, 100],
                'penalty': ['l1', 'l2']
            }
        }
        params = param_dict[str(self.clf)]
        if params is None:
            raise TypeError(
                'Expected [dict] but got None!\n'
                'Did you provide a valid classifier?'
            )
        return param_dict[str(self.clf)]

    def create(self):
        """
        Train the classifier with gridsearch and cross evaluation enabled.
        Return GSCV model. Warning: The returned model is not a classifier!
        """
        gs_model = GridSearchCV(self.clf, self.params, n_jobs=-1, cv=self.cv)
        gs_model.fit(self.X_train, self.y_train)
        self.gs_model = gs_model
        return self.gs_model

    def score(self):
        """
        Scores the best_estimator on the X_test and y_test
        datasets and returns it.
        """
        if not hasattr(self, 'gs_model'):
            self.fit()
        return self.gs_model.best_estimator_.score(self.X_test, self.y_test)

    def get_best_estimator(self):
        """
        Return the best_estimator object
        """
        if not hasattr(self, 'gs_model'):
            self.fit()
        return self.gs_model.best_estimator_

    def get_fit_details(self):
        """
        Print a table that shows the (training) scores
        for the various parameter combinations. 
        """
        if not hasattr(self, 'gs_model'):
            self.fit()
        for mean, std, params in zip(means, stds, self.gs_model.cv_results_['params']):  # noqa
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
