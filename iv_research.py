import numpy as np

from mergedict import load_all_pickle,get_implied_fee
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('ggplot')
mpl.rcParams['figure.figsize'] = [15, 9]

def demo1():
    # same (lower,upper) rate: different sigma and differnt fee
    lower = 0.8
    upper = 1.2

    sigma_list = np.arange(0.1, 1.3, 0.1)
    iv_Table = load_all_pickle()
    fee_list = [get_implied_fee(lower,upper,sigma,iv_Table) for sigma in sigma_list]
    # plot
    plt.plot(sigma_list,fee_list)
    plt.xlabel("sigma")
    plt.ylabel("implied fee")
    plt.title("implied fee vs sigma")
    plt.show()



