#####
#    Last update: Oct 5 2020
#    Author: Bahador Bahmani bb2969@columbia
#    Under supervision Prof. Waiching Sun
#####
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['text.usetex'] = True 
rcParams['text.latex.preamble'] = [
                                   r'\usepackage{amsmath}',
                                   r'\usepackage{amssymb}']
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Computer Modern'

plt.rcParams["figure.figsize"] = (10, 10/1.3)
plot_linestyle = ['-', '--', '-.', ':']
plot_linecolor = ['k', 'r', 'g', 'b', 'c', 'm']


def plot_x_y(x_all:list,y_all:list, x_label:str, y_label:str, legend_all=[None], add_to_save=None,
             xscale='', yscale='',plot_linestyle=plot_linestyle, plot_linecolor=plot_linecolor,
             xlim=None, ylim=None, add_leg=True, xticks_range=None, yticks_range=None,
             title=None, title_fontsize = None,leg_loc='best',
             need_return_plt=False):
    assert isinstance(x_all, list)
    assert isinstance(y_all, list)
    assert isinstance(legend_all, list)
    if need_return_plt:
        assert add_to_save==None
    for i, x in enumerate(x_all):
        plt.plot(x,y_all[i], label=legend_all[i], linewidth=3, linestyle = plot_linestyle[i],
                    color=plot_linecolor[i])
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.xticks(fontsize= 28)
    plt.yticks(fontsize= 28)
    plt.xlabel(x_label, fontsize=28)
    plt.ylabel(y_label, fontsize=28)
    if xscale:
        plt.xscale(xscale) #'log'
    if yscale:
        plt.yscale(yscale) #'log'
    if xticks_range is not None:
        assert xticks_range.ndim == 1 # np.arange(0, 91, 30.)
        plt.xticks(xticks_range)
    if yticks_range is not None:
        assert yticks_range.ndim == 1 # np.arange(0, 91, 30.)
        plt.yticks(yticks_range)
    if title:
       plt.title(title, fontsize=28)
    plt.grid(True)
    plt.tight_layout()
    if len(legend_all)>1 and add_leg:
        plt.legend(loc=leg_loc, prop={'size': 28})
    if add_to_save:
        plt.savefig(add_to_save)
        plt.close()
    else:
        if not need_return_plt:
            plt.show()
    if add_to_save == None:
        return plt
    else:
        return None