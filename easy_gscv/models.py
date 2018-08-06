"""
High level objects to automate large parts of the classifier training workflow.
"""
from typing import Optional, Dict, KeysView, Union
from sklearn.model_selection import (  # type: ignore
    train_test_split, GridSearchCV
)  # type: ignore
from sklearn.ensemble import (  # type: ignore
    RandomForestClassifier, GradientBoostingClassifier
    )  # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.neural_network import MLPClassifier  # type: ignore


# Create sklearn_classifiers Union object for mypy
ScikitClassifiers = Union[
    RandomForestClassifier, GradientBoostingClassifier,
    KNeighborsClassifier, LogisticRegression,
    MLPClassifier
]

# pylint: disable=too-many-instance-attributes
# pylint: disable-msg=too-many-arguments
# Arguments are needed to build the models.


class GSCV:
    """
    This object will build trained classifiers with the specified parameters.
    Training / testing datasets are created automatically from the
    [X,y] keywords and the models are trained using
    grid search and cross validation.
    """

    # Dictionary with all suported classifiers as values
    # and matching strings as keys
    model_dict = {
        'KNeighborsClassifier': KNeighborsClassifier(),
        'RandomForestClassifier': RandomForestClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier(),
        'MLPClassifier': MLPClassifier(),
        'LogisticRegression': LogisticRegression(),
    }

    def __init__(
            self, clf, X, y, cross_vals: int = 10, random_state: int = 42,
            test_size: float = 0.33, n_jobs: int = 1,
            params: Optional[Dict[str, Union[str, float]]] = None
    ) -> None:

        # Check if classifier is valid and assign it if true
        self.clf = self._get_model(clf)

        # Split data into Train and test sets
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.cross_vals = cross_vals
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Set params
        if params is None:
            params = self._get_model_params()
        self.params = params

        # Create model on initialization
        self.gs_model = self.create()

    @staticmethod
    def _get_model_name(clf, target='classifier') -> str:
        """
        With sklearn classifiers missing the '__name__' attribute,
        getting their name has proven to be suprisingly tricky.
        """
        if target == 'classifier':
            name = str(
                clf.__class__
            ).replace('>', '').split('.')[-1].replace("'", "")
        if target == 'class':
            name = str(clf.__class__).replace(
                "class '", ''
            ).replace('<', '').split('.')[0]
        return name

    def _get_model(
            self, clf: Union[str, ScikitClassifiers]
    ) -> ScikitClassifiers:
        """
        Fetches to correct model for the provided clf string
        or returns an error message if the clf is invalid.
        """

        error_message = """
        Model not supported! Please use one of the following classifiers:
        \n {}
        """.format(", ".join(map(str, self.model_dict)))

        # If a string is passed instead of a sklearn classifier
        # the program should check if the string matches a supported
        # classifier
        if isinstance(clf, str):
            if clf not in self.model_dict.keys():
                raise ValueError(error_message)
            return self.model_dict[clf]

        if self._get_model_name(clf, 'class') != 'sklearn':
            raise TypeError(
                'The classifier is not a valid scikit-learn classifier!'
            )

        if self._get_model_name(clf, 'classifier') \
                not in self.model_dict.keys():
            raise ValueError(error_message)

        name = self._get_model_name(clf, 'classifier')
        return self.model_dict[name]

    def _get_model_params(self) -> Dict:
        """
        An Internal method to get a matching model_param dict for the
        provided classifier, called only if no params are provided.
        """
        # Get number of features
        if self.x_train.shape[1] > 10:
            no_ftrs = self.x_train.shape[1]
        else:
            no_ftrs = 10

        # Choice of algorithm depends on size of dataset
        if self.x_train.shape[1] < 500:
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
        params = param_dict[self._get_model_name(self.clf, 'classifier')]
        return params

    @property
    def classifiers(self) -> KeysView[str]:
        """Returns a list of all valid classifiers"""
        return self.model_dict.keys()

    def create(self) -> GridSearchCV:
        """
        Train the classifier with gridsearch and cross evaluation enabled.
        Return GSCV model. Warning: The returned model is not a classifier!
        """
        gs_model = GridSearchCV(
            self.clf, self.params, n_jobs=self.n_jobs, cv=self.cross_vals
        )
        gs_model.fit(self.x_train, self.y_train)
        return gs_model

    def score(self) -> float:
        """
        Scores the best_estimator on the X_test and y_test
        datasets and returns it.
        """
        return self.gs_model.best_estimator_.score(self.x_test, self.y_test)

    def get_best_estimator(self) -> object:
        """
        Return the best_estimator object
        """
        return self.gs_model.best_estimator_

    def get_fit_details(self) -> str:
        """
        Print a table that shows the (training) scores
        for the various parameter combinations.
        """
        table = ""
        means = self.gs_model.cv_results_['mean_test_score']
        stds = self.gs_model.cv_results_['std_test_score']
        for mean, std, params in zip(
                means, stds, self.gs_model.cv_results_['params']
        ):
            table += "%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params)
        return table
