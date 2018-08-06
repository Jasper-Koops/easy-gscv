"""
FIXME module add docstrings
"""
from unittest import TestCase, main
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from easy_gscv.models import GSCV

# pylint: disable=C0103
# 'x' and 'y' are very descriptive in the realm of datascience.

# pylint: disable=W0212
# Method call required for testing


class TestProperties(TestCase):
    """
    Make all model properties work as they should.
    """

    def setUp(self):
        # Get data
        from sklearn import datasets
        iris = datasets.load_iris()
        self.x = iris['data']
        self.y = iris['target']
        self.clf = KNeighborsClassifier()
        self.model = GSCV(self.clf, self.x, self.y)
        self.model_dict = {
            'KNeighborsClassifier': KNeighborsClassifier(),
            'RandomForestClassifier': RandomForestClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'MLPClassifier': MLPClassifier(),
            'LogisticRegression': LogisticRegression(),
        }

    def test_classifiers(self):
        """
        Test that the 'classifiers' property returns
        a list of valid classifiers
        """
        self.assertEqual(
            self.model.classifiers, self.model_dict.keys())


class TestCLFTypes(TestCase):
    """
    Test that the clf argument can handle multiple types.
    """

    def setUp(self):
        # Get data
        from sklearn import datasets
        iris = datasets.load_iris()
        self.x = iris['data']
        self.y = iris['target']

    def test_sklearn_classifier(self):
        """
        Test that object accepts a sklearn classifier
        """
        clf = KNeighborsClassifier()
        model = GSCV(clf, self.x, self.y)
        self.assertEqual(
            type(model._get_model(clf)), type(KNeighborsClassifier())
        )

    def test_string(self):
        """
        Test that object accepts a string
        """
        clf = 'KNeighborsClassifier'
        model = GSCV(clf, self.x, self.y)
        self.assertEqual(
            type(model._get_model(clf)), type(KNeighborsClassifier())
        )


class TestExceptions(TestCase):
    """
    Make sure that the exceptions trigger when they should.
    """

    def setUp(self):
        # Get data
        from sklearn import datasets
        iris = datasets.load_iris()
        self.x = iris['data']
        self.y = iris['target']

        class Nothing:
            """Nonsense object that should not pass"""
            def __init__(self, a):
                self.a = a

        self.wrong_object = Nothing(a='horse')
        self.wrong_clf = DecisionTreeClassifier()
        self.valid_clf = KNeighborsClassifier()

    def test_check_model_not_a_model(self):
        """
        Make sure that using a type that is not a model
        raises the correct error message.
        """
        with self.assertRaises(ValueError):
            GSCV('doesnotexist', self.x, self.y)

    def test_check_model_not_a_sklearn_model(self):
        """
        Make sure that using a non-sklearn model raises
        the correct error message.
        """
        with self.assertRaises(TypeError):
            GSCV(self.wrong_object, self.x, self.y)

    def test_check_wrong_scikit_model(self):
        """
        Make sure that using a non-valid sklearn classifier
        raises the correct error message
        """
        with self.assertRaises(ValueError):
            GSCV(self.wrong_clf, self.x, self.y)


class TestKNeighborsClassifier(TestCase):
    """
    Make sure that the model is created and the methods work
    """

    def setUp(self):
        # Get data
        from sklearn import datasets
        iris = datasets.load_iris()
        self.x = iris['data']
        self.y = iris['target']
        self.valid_clf = KNeighborsClassifier()

    def test_default_params(self):
        """
        Make sure that the correct default parameters
        are selected for the model
        """
        model = GSCV(self.valid_clf, self.x, self.y)
        self.assertEqual(
            model.params, {
                'n_neighbors': [3, 5, 8, 10, 15],
                'weights': ['uniform', 'distance'],
            }
        )

    def test_custom_params(self):
        """Test that custom params override the default ones"""
        model = GSCV(self.valid_clf, self.x, self.y, params={
            'n_neighbors': [3, 15],
            'weights': ['uniform'],
        })
        self.assertEqual(
            model.params, {
                'n_neighbors': [3, 15],
                'weights': ['uniform'],
            }
        )

    def test_create(self):
        """Test that the create method returns a value"""
        model = GSCV(self.valid_clf, self.x, self.y)
        result = model.create()
        self.assertTrue(result is not None)

    def test_score(self):
        """Test that the create method returns a value"""
        model = GSCV(self.valid_clf, self.x, self.y)
        score = model.score()
        self.assertTrue(score is not None)
        self.assertTrue(0 <= score <= 1)

    def test_get_best_estimator(self):
        """Test that the 'get_best_estimator' method returns a value"""
        model = GSCV(self.valid_clf, self.x, self.y)
        best_model = model.get_best_estimator()
        self.assertTrue(best_model is not None)

    def test_get_fit_details(self):
        """Test that the 'get_fit_details' method returns a value"""
        model = GSCV(self.valid_clf, self.x, self.y)
        fit_details = model.get_fit_details()
        self.assertTrue(fit_details is not None)


class TestLogisticRegression(TestCase):
    """ Make sure that the model is created and the methods work """

    def setUp(self):
        # Get data
        from sklearn import datasets
        iris = datasets.load_iris()
        self.x = iris['data']
        self.y = iris['target']
        self.valid_clf = LogisticRegression()

    def test_default_params(self):
        """
        Make sure that the correct default parameters
        are selected for the model
        """
        model = GSCV(self.valid_clf, self.x, self.y)
        self.assertEqual(
            model.params, {
                'C': [0.01, 1, 100],
                'penalty': ['l1', 'l2']
            }
        )

    def test_custom_params(self):
        """Test that custom params override the default ones"""
        model = GSCV(self.valid_clf, self.x, self.y, params={
            'C': [1, 100],
            'penalty': ['l2']
        })
        self.assertEqual(
            model.params, {
                'C': [1, 100],
                'penalty': ['l2']
            }
        )

    def test_create(self):
        """
        Test that the create method returns a value
        """
        model = GSCV(self.valid_clf, self.x, self.y)
        result = model.create()
        self.assertTrue(result is not None)

    def test_score(self):
        """
        Test that the create method returns a value
        """
        model = GSCV(self.valid_clf, self.x, self.y)
        score = model.score()
        self.assertTrue(score is not None)
        self.assertTrue(0 <= score <= 1)

    def test_get_best_estimator(self):
        """
        Test that the 'get_best_estimator' method returns a value
        """
        model = GSCV(self.valid_clf, self.x, self.y)
        best_model = model.get_best_estimator()
        self.assertTrue(best_model is not None)

    def test_get_fit_details(self):
        """
        Test that the 'get_fit_details' method returns a value
        """
        model = GSCV(self.valid_clf, self.x, self.y)
        fit_details = model.get_fit_details()
        self.assertTrue(fit_details is not None)


class TestMLPClassifier(TestCase):
    """
    Make sure that the model is created and the methods work
    """

    def setUp(self):
        # Get data
        from sklearn import datasets
        iris = datasets.load_iris()
        self.x = iris['data']
        self.y = iris['target']
        self.valid_clf = MLPClassifier()

    def test_default_params(self):
        """
        Make sure that the correct default parameters
        are selected for the model
        """
        model = GSCV(self.valid_clf, self.x, self.y)
        self.assertEqual(
            model.params, {
                'hidden_layer_sizes': [
                    (10,),
                    (10, 10),
                    (10, 10, 10)
                ],
                'alpha': [0.0001, 0.01, 0.1, 1],
                'solver': ['lbfgs'],
            }
        )

    def test_custom_params(self):
        """Test that custom params override the default ones"""
        model = GSCV(self.valid_clf, self.x, self.y, params={
            'alpha': [0.0001, 0.01, 0.1, 1]
        })
        self.assertEqual(
            model.params, {
                'alpha': [0.0001, 0.01, 0.1, 1]
            }
        )

    def test_create(self):
        """
        Test that the create method returns a value
        """
        model = GSCV(self.valid_clf, self.x, self.y, params={
            'alpha': [0.0001, 0.01, 0.1, 1]
        })
        result = model.create()
        self.assertTrue(result is not None)

    def test_score(self):
        """
        Test that the create method returns a value
        """
        model = GSCV(self.valid_clf, self.x, self.y, params={
            'alpha': [0.0001, 0.01, 0.1, 1]
        })
        score = model.score()
        self.assertTrue(score is not None)
        self.assertTrue(0 <= score <= 1)

    def test_get_best_estimator(self):
        """
        Test that the 'get_best_estimator' method returns a value
        """
        model = GSCV(self.valid_clf, self.x, self.y, params={
            'alpha': [0.0001, 0.01, 0.1, 1]
        })
        best_model = model.get_best_estimator()
        self.assertTrue(best_model is not None)

    def test_get_fit_details(self):
        """
        Test that the 'get_fit_details' method returns a value
        """
        model = GSCV(self.valid_clf, self.x, self.y, params={
            'alpha': [0.0001, 0.01, 0.1, 1]
        })
        fit_details = model.get_fit_details()
        self.assertTrue(fit_details is not None)


class TestRandomForestClassifier(TestCase):
    """
    Make sure that the model is created and the methods work
    """

    def setUp(self):
        # Get data
        from sklearn import datasets
        iris = datasets.load_iris()
        self.x = iris['data']
        self.y = iris['target']
        self.valid_clf = RandomForestClassifier()

    # def test_default_params(self):
    #     model = GSCV(self.valid_clf, self.x, self.y, n_jobs=-1)
    #     self.assertEqual(
    #         model.params, {
    #             'n_estimators': [100, 500, 1000],
    #             'max_features': ['sqrt', 'log2', None],
    #             'max_depth': [None, 3, 5],
    #         }
    #     )

    def test_custom_params(self):
        """Test that custom params override the default ones"""
        model = GSCV(self.valid_clf, self.x, self.y, n_jobs=-1, params={
            'n_estimators': [100],
            'max_features': ['sqrt'],
            'max_depth': [3],
        })
        self.assertEqual(
            model.params, {
                'n_estimators': [100],
                'max_features': ['sqrt'],
                'max_depth': [3],
            }
        )

    def test_create(self):
        """
        Test that the create method returns a value
        """
        model = GSCV(self.valid_clf, self.x, self.y, n_jobs=-1, params={
            'n_estimators': [100],
            'max_features': ['sqrt'],
            'max_depth': [3],
        })
        result = model.create()
        self.assertTrue(result is not None)

    def test_score(self):
        """
        Test that the create method returns a value
        """
        model = GSCV(self.valid_clf, self.x, self.y, n_jobs=-1, params={
            'n_estimators': [100],
            'max_features': ['sqrt'],
            'max_depth': [3],
        })
        score = model.score()
        self.assertTrue(score is not None)
        self.assertTrue(0 <= score <= 1)

    def test_get_best_estimator(self):
        """
        Test that the 'get_best_estimator' method returns a value
        """
        model = GSCV(self.valid_clf, self.x, self.y, n_jobs=-1, params={
            'n_estimators': [100],
            'max_features': ['sqrt'],
            'max_depth': [3],
        })
        best_model = model.get_best_estimator()
        self.assertTrue(best_model is not None)

    def test_get_fit_details(self):
        """
        Test that the 'get_fit_details' method returns a value
        """
        model = GSCV(self.valid_clf, self.x, self.y, n_jobs=-1, params={
            'n_estimators': [100],
            'max_features': ['sqrt'],
            'max_depth': [3],
        })
        fit_details = model.get_fit_details()
        self.assertTrue(fit_details is not None)


class TestGradientBoostingClassifier(TestCase):
    """
    Make sure that the model is created and the methods work
    """

    def setUp(self):
        # Get data
        from sklearn import datasets
        iris = datasets.load_iris()
        self.x = iris['data']
        self.y = iris['target']
        self.valid_clf = GradientBoostingClassifier()

    # def test_default_params(self):
    #     model = GSCV(self.valid_clf, self.x, self.y, n_jobs=-1)
    #     self.assertEqual(
    #         model.params, {
    #             'n_estimators': [100, 500, 1000],
    #             'learning_rate': [0.1, 0.5, 1],
    #             'max_depth': [1, 3, 5],
    #             'max_features': ['sqrt', 'log2', None],
    #         }
    #     )

    def test_custom_params(self):
        """Test that custom params override the default ones"""
        model = GSCV(self.valid_clf, self.x, self.y, n_jobs=-1, params={
            'n_estimators': [100],
            'learning_rate': [0.1, 0.5],
            'max_depth': [1, 3],
            'max_features': ['sqrt'],
        })
        self.assertEqual(
            model.params, {
                'n_estimators': [100],
                'learning_rate': [0.1, 0.5],
                'max_depth': [1, 3],
                'max_features': ['sqrt'],
            }
        )

    def test_create(self):
        """Test that the create method returns a value"""
        model = GSCV(self.valid_clf, self.x, self.y, n_jobs=-1, params={
            'n_estimators': [100],
            'learning_rate': [0.1],
            'max_depth': [1],
            'max_features': ['sqrt'],
        })
        result = model.create()
        self.assertTrue(result is not None)

    def test_score(self):
        """Test that the create method returns a value"""
        model = GSCV(self.valid_clf, self.x, self.y, n_jobs=-1, params={
            'n_estimators': [100],
            'learning_rate': [0.1],
            'max_depth': [1],
            'max_features': ['sqrt'],
        })
        score = model.score()
        self.assertTrue(score is not None)
        self.assertTrue(0 <= score <= 1)

    def test_get_best_estimator(self):
        """Test that the 'get_best_estimator' method returns a value"""
        model = GSCV(self.valid_clf, self.x, self.y, n_jobs=-1, params={
            'n_estimators': [100],
            'learning_rate': [0.1],
            'max_depth': [1],
            'max_features': ['sqrt'],
        })
        best_model = model.get_best_estimator()
        self.assertTrue(best_model is not None)

    def test_get_fit_details(self):
        """Test that the 'get_fit_details' method returns a value"""
        model = GSCV(self.valid_clf, self.x, self.y, n_jobs=-1, params={
            'n_estimators': [100],
            'learning_rate': [0.1],
            'max_depth': [1],
            'max_features': ['sqrt'],
        })
        fit_details = model.get_fit_details()
        self.assertTrue(fit_details is not None)


if __name__ == '__main__':
    print('\n\n')
    print("Running tests!")
    print(
        'Warning: Despite the use of a (relatively) test small dataset '
        'these tests can take up to 2 minutes to complete on a relativly '
        'modern computer due to the number of models being trained.'
        'The models have been set up to utilize all available cpu cores '
        'to speed up the process and the more computationally models have '
        'their "test_default_params" method tests disabled, while their '
        'other tests use lighter-than-default params.'
        '\n\nResetting these values to their default parameters '
        'will increase the runtime significantly'
        )
    print('\n\n')
    main()
