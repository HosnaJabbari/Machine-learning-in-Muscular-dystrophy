from enum import Enum
import abc
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


class Classifier_type(Enum):
    LINEAR_REGRESSION = 1
    SVR_RBF = 2
    SVR_LINEAR = 3
    SVR_POLYNOMIAL = 4
    KERNEL_RIDGE = 5
    KNEIGHBORS_REGRESSOR = 6
    DECISION_TREE_REGRESSOR = 7

#class Classifier(abc.ABC):
class Classifier:
    __metaclass__ = abc.ABCMeta
    """Defining an abstract base class to represent the API of a set of classifiers"""

    @abc.abstractmethod
    def __init__(self):
        """"This init function must be called by the subclasses."""
        self.classifier = None

    @abc.abstractmethod 
    def instantiate(self):
        """Get a valid classifier type and return an instance of that classifier (also the classifier should be assigned
         to self.classifier variable/property). Valid classifier types are defined in the Classifier_type enumeration
         class."""
        return
         
    @abc.abstractmethod
    def represent(self):
        """First check if the classifier has been instantiated. Then, represent a classifier using the information
        obtained from fitting a model to the data. Return the model details thoroughly"""
        return

    def isExist_classifier(self):
        if self.classifier is None:
            'No classifier is instanciated. Please use instantiate() first.'
            return False
        return True


class Linear_regression(Classifier, LinearRegression):
    
    def __init__(self):
        super(Linear_regression, self).__init__()

    def instantiate(self):
        self.classifier = LinearRegression()  # Create linear regression object
        return self.classifier

    def represent(self):
        if not self.isExist_classifier():
            return
      
        coef = ('Coefficients: ', self.classifier.coef_)
        intercept = ('Intercept: ', self.classifier.intercept_)
        return [coef, intercept]


class KNeighbors_regressor(Classifier, KNeighborsRegressor):
    
    def __init__(self, neighbors_num):
        super(KNeighbors_regressor, self).__init__()
        self.neighbors_num = neighbors_num
    
    def instantiate(self):
        self.classifier = KNeighborsRegressor(n_neighbors=self.neighbors_num)
        return self.classifier
    
    
    def represent(self):
        if not self.isExist_classifier():
            return
        representation = 'No specific extra info'
        return [representation]


class Decision_tree_regressor(Classifier, DecisionTreeRegressor):
    
    def __init__(self):
        super(Decision_tree_regressor, self).__init__()
    
    
    def instantiate(self):
        self.classifier = DecisionTreeRegressor(random_state=0)
        return self.classifier
    
    
    def represent(self):
        if not self.isExist_classifier():
            return
      
        feature_importances = ('feature_importances (Gini importance): ', self.classifier.feature_importances_)
        intercept = ('max_features: ', self.classifier.max_features_)
        n_features = ('n_features: ', self.classifier.n_features_)
        n_outputs = ('n_outputs: ', self.classifier.n_outputs_ )
        tree = ('tree: ', self.classifier.tree_)
        return [feature_importances, intercept, n_features, n_outputs, tree]


class SVR_linear(Classifier, SVR):

    def __init__(self):
        super(SVR_linear, self).__init__()


    def instantiate(self):
        self.classifier = SVR(kernel='linear', C=1e3)
        return self.classifier


    def represent(self):
        if not self.isExist_classifier():
            return

        support  = ('support: ', self.classifier.support_ )
        support_vectors = ('support_vectors: ', self.classifier.support_vectors_)
        dual_coef = ('dual_coef: ', self.classifier.dual_coef_)
        coef = ('Coefficients: ', self.classifier.coef_)
        intercept = ('Intercept: ', self.classifier.intercept_)
        sample_weight = ('sample_weight: ', self.classifier.sample_weight) 
        return [support, support_vectors, dual_coef, coef, intercept, sample_weight]


class SVR_RBF(Classifier, SVR):

    def __init__(self):
        super(SVR_RBF, self).__init__()


    def instantiate(self):
        self.classifier = SVR(kernel='rbf', C=1e3, gamma=0.1)
        return self.classifier

    def represent(self):
        if not self.isExist_classifier():
            return

        support  = ('support: ', self.classifier.support_ )
        support_vectors = ('support_vectors: ', self.classifier.support_vectors_)
        dual_coef = ('dual_coef: ', self.classifier.dual_coef_)

        intercept = ('Intercept: ', self.classifier.intercept_)
        sample_weight = ('sample_weight: ', self.classifier.sample_weight) 
        return [support, support_vectors, dual_coef, intercept, sample_weight]


class SVR_polynomial(Classifier, SVR):

    def __init__(self):
        super(SVR_polynomial, self).__init__()


    def instantiate(self):
        self.classifier = SVR(kernel='poly', C=1e3, degree=2)
        return self.classifier

    def represent(self):
        if not self.isExist_classifier():
            return

        support  = ('support: ', self.classifier.support_ )
        support_vectors = ('support_vectors: ', self.classifier.support_vectors_)
        dual_coef = ('dual_coef: ', self.classifier.dual_coef_)

        intercept = ('Intercept: ', self.classifier.learner.intercept_)
        sample_weight = ('sample_weight: ', self.classifier.sample_weight) 
        return [support, support_vectors, dual_coef, intercept, sample_weight]


class Kernel_ridge(Classifier, KernelRidge):

    def __init__(self):
        super(Kernel_ridge, self).__init__()


    def instantiate(self):
        self.classifier = KernelRidge(alpha=1.0)
        return self.classifier


    def represent(self):
        if not self.isExist_classifier():
            return

        dual_coef = ('dual_coef: ', self.classifier.dual_coef_)
        X_fit = ('X_fit: ', self.classifier.X_fit_)

        return [dual_coef, X_fit]

