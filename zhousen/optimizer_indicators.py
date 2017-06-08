#import operator
import numpy as np
#import math

def get_optimizer_indicators(weight, cov_matrix):

    weight = np.array(weight)
    # refer to paper formula 2.19 2.20
    production_i = weight * ( cov_matrix.dot(weight) )
    productions = weight.dot(cov_matrix).dot(weight)

    # calculate hightest risk contributions
    HRCs = production_i / productions
    #index, value = max(enumerate(HRCs), key=operator.itemgetter(1))
    value, index = max([(v, i) for i, v in enumerate(HRCs)])

    HRC = value
    HRC_index = index

    # calculate Herfindahl
    Herfindahl = np.sum(HRCs**2)

    return(HRC, HRC_index, Herfindahl)



##  ex

# cov_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 4]])
# weight = np.array([0.5, 0.35, 0.15])
# weight = [0.5, 0.333, 0.167]

# get_optimizer_indicators(weight , cov_matrix)





