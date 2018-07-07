from statistics import mean
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style

import random

style.use('ggplot')

def create_dataset(howmuch, variance, step = 2, correlation = False):
    val = 1
    ys = []
    for i in range(howmuch):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step

    xs = [i for i in range(len(ys))]
    
    return np.array(xs, dtype = np.float64), np.array(ys, dtype = np.float64)

def best_fit_slope_and_intercept(xs, ys):

    m = (((mean(xs) * mean(ys)) - mean(xs*ys)) / ((mean(xs)**2) - mean(xs*xs)))
    b = mean(ys) - m*mean(xs)
    
    return m,b

def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_mean)

def main():
##    xs = [1,2,3,4,5]
##    ys = [5,4,6,5,6]
##
##    xs = np.array([1,2,3,4,5], dtype = np.float64)
##    ys = np.array([5,4,6,5,6], dtype = np.float64)

    xs, ys = create_dataset(40, 10, 2, correlation = 'pos')
    
    m,b = best_fit_slope_and_intercept(xs, ys)
    print(m, b)

    regression_line = [(m*x) + b for x in xs]

##    predict_x = 7
##    predict_y = (m*predict_x) + b
##    print(predict_y)

    r_squared = coefficient_of_determination(ys, regression_line)
    print(r_squared)

    plt.scatter(xs, ys, color = '#003F72', label = 'data')
    plt.plot(xs, regression_line, label = 'regression line')
##    plt.scatter(predict_x, predict_y, color = '#2eb82e', label = 'prediction')
    plt.legend(loc=4)
    plt.show()

if __name__ == '__main__':
    main()


