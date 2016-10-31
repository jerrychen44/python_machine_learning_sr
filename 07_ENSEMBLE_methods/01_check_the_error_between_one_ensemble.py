

from scipy.misc import comb
import math
import numpy as np

import matplotlib.pyplot as plt


# the probability mass function of binomial distribution
# ref: https://en.wikipedia.org/wiki/Binomial_distribution
# search :probability mass function
def ensemble_error(n_classifier, error):
    k_start = math.ceil(n_classifier / 2.0)
    probs = [comb(n_classifier, k) * error**k *
               (1-error)**(n_classifier - k)
             for k in range(k_start, n_classifier + 1)]
    return sum(probs)







def check_the_error_between_one_ensemble():


    # we can check the error between one classifer and ensemble classifer
    # use the range 0~1 error range.
    error_range = np.arange(0.0, 1.01, 0.01)
    ens_errors = [ensemble_error(n_classifier=11,
                           error=error)
                for error in error_range]




    #plot out
    plt.plot(error_range,
             ens_errors,
             label='Ensemble error',
             linewidth=2)

    plt.plot(error_range,
             error_range,
             linestyle='--',
             label='Base error',
             linewidth=2)

    plt.xlabel('Base error')
    plt.ylabel('Base/Ensemble error')
    plt.legend(loc='upper left')
    plt.grid()
    #plt.tight_layout()
    # plt.savefig('./figures/ensemble_err.png', dpi=300)
    plt.show()

    return 0


def main():

    check_the_error_between_one_ensemble()




    return 0

main()
