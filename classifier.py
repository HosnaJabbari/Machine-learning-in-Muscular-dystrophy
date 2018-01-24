from enum import Enum
import abc
from sklearn.neighbors import KNeighborsRegressor
from sklearn import model_selection, linear_model
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

# Defining an abstract base class to represent the API of a set of classifiers
class Classifier(abc.ABC):
#class Classifier(metaclass=abc.ABCMeta):
   
    @abc.abstractmethod
    def __init__(self, classifier_type):
        """"This init function must be called by the subclasses."""
        self.type =  classifier_type # TODO check if it blongs to only the parent class (i.e. all instances) or the instsnce
        self.classifier = None

    @abc.abstractmethod 
    def instanciate(self, classifier_type): 
        """Get a valid classifier type and return an instance of that classifier (also the classifier should be assigned to self.classifier variable/property). Valid classifier types are defined in the Classifier_type enumeratoin class."""
        return
         
    @abc.abstractmethod
    def represent(self):
        """First check if the classifier has been instanciated. Then, represent a classifier using the information obtained from fitting a model to the data. Return the model details throughly"""
        return

    def isExist_classifier():
        if self.classifier == None:
            'No classifier is instanciated. Please use instanciate(classifier_type) first.'
            return False
        return True


class Linear_regression(Classifier):
    
    def __init__(self, classifier_type):
        super(Linear_regression, self),__inint__(classifier_type)
    
    
    def instanciate(self, classifier_type):
        self.classifier = linear_model.LinearRegression() # Create linear regression object
        return self.classifier
    
    
    def def represenit(self):
        if not isExist_classifier():
            return
      
        coef = ('Coefficients: ', self.classifier.coef_)
        intercept = ('Intercept: ', self.classifier.intercept_)
        return [coef, intercept]



class KNeighbors_regressor(Classifier):
    
    def __init__(self, classifier_type, neighbors_num):
        super(Linear_regression, self),__inint__(classifier_type)
        self.neighbors_num = neighbors_num
    
    def instanciate(self, classifier_type):
        self.classifier = NearestNeighbors(n_neighbors = self.neighbors_num)
        return self.classifier
    
    
    def def represenit(self):
        if not isExist_classifier():
            return
        representation = 'No specific extra info'
        return [representation]



class Decision_tree_regressor(Classifier):
    
    def __init__(self, classifier_type):
        super(Linear_regression, self),__inint__(classifier_type)
    
    
    def instanciate(self, classifier_type):
        self.classifier = DecisionTreeRegressor(random_state=0)
        return self.classifier
    
    
    def def represenit(self):
        if not isExist_classifier():
            return
      
        feature_importances = ('feature_importances (Gini importance): ', self.classifier.feature_importances_)
        intercept = ('max_features: ', self.classifier.max_features_)
        n_features = ('n_features: ', self.classifier.n_features_)
        n_outputs = ('n_outputs: ', self.classifier.n_outputs_ )
        tree = ('tree: ', self.classifier.tree_)
        return [feature_importances, intercept, n_features, n_outputs, tree]



class SVR_linear(Classifier):

    def __init__(self, classifier_type):
        super(Linear_regression, self),__inint__(classifier_type)


    def instanciate(self, classifier_type):
        self.classifier = SVR(kernel='linear', C=1e3)
        return self.classifier


    def def represenit(self):
        if not isExist_classifier():
            return

        support  = ('support: ', self.classifier.support_ )
        support_vectors = ('support_vectors: ', self.classifier.support_vectors_)
        dual_coef = ('dual_coef: ', self.classifier.dual_coef_)
        coef = ('Coefficients: ', self.classifier.coef_)
        intercept = ('Intercept: ', self.classifier.intercept_)
        sample_weight = ('sample_weight: ', self.classifier.sample_weight) 
        return [support, support_vectors, dual_coef, coef, intercept, sample_weight]



class SVR_RBF(Classifier):

    def __init__(self, classifier_type):
        super(Linear_regression, self),__inint__(classifier_type)


    def instanciate(self, classifier_type):
        self.classifier = SVR(kernel='rbf', C=1e3, gamma=0.1)
        return self.classifier

    def def represenit(self):
        if not isExist_classifier():
            return

        support  = ('support: ', self.classifier.support_ )
        support_vectors = ('support_vectors: ', self.classifier.support_vectors_)
        dual_coef = ('dual_coef: ', self.classifier.dual_coef_)

        intercept = ('Intercept: ', self.classifier.intercept_)
        sample_weight = ('sample_weight: ', self.classifier.sample_weight) 
        return [support, support_vectors, dual_coef, intercept, sample_weight]



class SVR_polynomial(Classifier):

    def __init__(self, classifier_type):
        super(Linear_regression, self),__inint__(classifier_type)


    def instanciate(self, classifier_type):
        self.classifier = SVR(kernel='poly', C=1e3, degree=2)
        return self.classifier

    def def represenit(self):
        if not isExist_classifier():
            return

        support  = ('support: ', self.classifier.support_ )
        support_vectors = ('support_vectors: ', self.classifier.support_vectors_)
        dual_coef = ('dual_coef: ', self.classifier.dual_coef_)

        intercept = ('Intercept: ', self.classifier.learner.intercept_)
        sample_weight = ('sample_weight: ', self.classifier.sample_weight) 
        return [support, support_vectors, dual_coef, intercept, sample_weight]




class kernel_ridge(Classifier):

    def __init__(self, classifier_type):
        super(Linear_regression, self),__inint__(classifier_type)


    def instanciate(self, classifier_type):
        self.classifier = KernelRidge(alpha=1.0)
        return self.classifier


    def def represenit(self):
        if not isExist_classifier():
            return

        dual_coef = ('dual_coef: ', self.classifier.dual_coef_)
        X_fit = ('X_fit: ', self.classifier.X_fit_)

        return [dual_coef, X_fit]

