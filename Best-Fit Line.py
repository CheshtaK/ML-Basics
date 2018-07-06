from statistics import mean
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

def best_fit_slope_and_intercept(xs, ys):

    m = (((mean(xs) * mean(ys)) - mean(xs*ys)) / ((mean(xs)**2) - mean(xs*xs)))
    b = mean(ys) - m*mean(xs)
    
    return m,b

def main():
    xs = [1,2,3,4,5]
    ys = [5,4,6,5,6]

    xs = np.array([1,2,3,4,5], dtype = np.float64)
    ys = np.array([5,4,6,5,6], dtype = np.float64)

    m,b = best_fit_slope_and_intercept(xs, ys)
    print(m, b)

    regression_line = [(m*x) + b for x in xs]

    predict_x = 7
    predict_y = (m*predict_x) + b
    print(predict_y)

    plt.scatter(xs, ys, color = '#003F72', label = 'data')
    plt.plot(xs, regression_line, label = 'regression line')
    plt.scatter(predict_x, predict_y, color = '#2eb82e', label = 'prediction')
    plt.legend(loc=4)
    plt.show()

if __name__ == '__main__':
    main()


