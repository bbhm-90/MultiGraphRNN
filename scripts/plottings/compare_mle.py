#####
#    Last update: Sep 28 2020
#    Author: bb2969@columbia
#####
import os
import pandas as pd
import matplotlib.pyplot as plt
root_folder = './results/'
folder_to_save = './results/figures_mle/'
mse_file_name = 'data_postproc/mse_over_all_data_sets.csv'
ann_cases= ['ANN_graphs_bulk_plasticity_classical_without_fabric',
            'ANN_graphs_bulk_plasticity_classical_with_fabric',
            'ANN_graphs_bulk_plasticity_new_fabric_with_classical_graph',
            ]
ann_graph_name = ['{eps->phi},{eps,phi->sig}',
                  '{eps->phi},{eps,phi->F},{eps,phi,F->sig}',
                  '{eps->phi},{topo},{eps,topo,phi->F}, {eps, phi, F->sig}',
                  ]
target_fields = ['sig11', 'sig22', 'sig33', 'sig12', 'sig23', 'sig13']
def plot1():
    plt.rcParams["figure.figsize"] = (9, 9/1.3)
    for case in ann_cases:
        add = root_folder + case + '/' + mse_file_name
        mse_info = pd.read_csv(add, delimiter=',')
        for f in target_fields:
            plt.plot(mse_info[f])
            plt.ylabel('mse {}'.format(f), fontsize=22)
            plt.legend(loc="best")
            # root_folder = './results/figures_ANN_graphs_bulk_plasticity_new_fabric_with_classical_graph/ave_{}_to_{}/'.format(init_set_id, end_set_id)
            # if not os.path.exists(root_folder): os.system("mkdir {}".format(root_folder))
            # plt.savefig(root_folder + '{}.pdf'.format(key))
            plt.show()
            # plt.close()
        exit()

def plot2():
    plt.rcParams["figure.figsize"] = (9, 9/1.3)
    if not os.path.exists(folder_to_save): os.system("mkdir {}".format(folder_to_save))
    for f in target_fields:
        for i, case in enumerate(ann_cases):
            add = root_folder + case + '/' + mse_file_name
            mse_info = pd.read_csv(add, delimiter=',')
            plt.plot(mse_info[f], label=ann_graph_name[i])
        plt.ylabel('mle {}'.format(f), fontsize=22)
        plt.xlabel('data set id'.format(f), fontsize=22)
        plt.legend(loc="best")
        plt.savefig(folder_to_save + '{}.png'.format(f))
        # plt.show()
        plt.close()
plot2()