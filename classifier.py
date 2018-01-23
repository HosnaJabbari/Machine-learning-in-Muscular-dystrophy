from enum import Enum
import abc


class Classifier_type(Enum):
    LINEAR_REGRESSION = 1
    SVR_RBF = 2
    SVR_LINEAR = 3
    SVR_POLYNOMIAL = 4
    KERNEL_RIDGE = 5


# Defining an abstract base class to represent the API of a set of classifiers
class Classifier:
   
    def __init__(self):
        self.type = None 

    
    def instanciate(self, classifier_type): 
    


    def represent(self):
        #  TODO
