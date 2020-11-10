#####
#    Last update: Sep 28 2020
#    Author: bb2969@columbia
#####
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 10/1.3)
root_folder = './results/'
ann_cases= ['ANN_graphs_bulk_plasticity_classical_without_fabric',
            #'ANN_graphs_bulk_plasticity_classical_with_fabric',
            #'ANN_graphs_bulk_plasticity_new_fabric_with_classical_graph',
            ]
data_sets_to_analyze = np.array(range(59))
num_loads = 501
plot_linestyle = ['-', '--', '-.']
plot_linecolor = ['k', 'b', 'g', 'r', 'c', 'm']


def get_stress_tensor(data_df, row_id):
    sig = np.zeros((3,3))
    sig[0,0] = data_df['sig11'].values[row_id]
    sig[1,1] = data_df['sig22'].values[row_id]
    sig[2,2] = data_df['sig33'].values[row_id]
    sig[0,1] = sig[1,0] = data_df['sig12'].values[row_id]
    sig[1,2] = sig[2,1] = data_df['sig23'].values[row_id]
    sig[0,2] = sig[2,0] = data_df['sig13'].values[row_id]
    return sig
def get_p_q(sig):
    p = np.trace(sig)/3.
    s = sig - p *np.eye(3)
    q = np.float(np.tensordot(s, s))
    q *= np.sqrt(3/2)
    return [p, q]
def plot_x_y(x_all,y_all, x_label, y_label, legend_all=[None], add_to_save=None):
    assert isinstance(x_all, list)
    assert isinstance(y_all, list)
    assert isinstance(legend_all, list)
    for i, x in enumerate(x_all):
        plt.plot(x,y_all[i], label=legend_all[i], linewidth=3, linestyle = plot_linestyle[i],
                    color=plot_linecolor[i])
    plt.xticks(fontsize= 22)
    plt.yticks(fontsize= 22)
    plt.xlabel(x_label, fontsize=22)
    plt.ylabel(y_label, fontsize=22)
    plt.grid(True)
    plt.tight_layout()
    if len(legend_all)>1:
        plt.legend(loc="best", prop={'size': 18})
    if add_to_save:
        plt.savefig(add_to_save)
        plt.close()
    else:
        plt.show()

q123_exp = np.zeros((num_loads, 3))
q123_pred = np.zeros((num_loads, 3))
pq_exp = np.zeros((num_loads,2))
pq_pred = np.zeros((num_loads,2))
for i, case in enumerate(ann_cases):
    for data_set in data_sets_to_analyze:
        folder_add = root_folder + case +'/data_postproc/dataset_{}/'.format(data_set)
        exp_data = pd.read_csv(folder_add + 'data_experiment.csv', delimiter=',')
        pred_data = pd.read_csv(folder_add + 'data_prediction_ave_over_200.csv', delimiter=',')
        features_pred = list(pred_data.columns)
        eps_11 = exp_data['eps11'].values
        phi_exp = exp_data['poro'].values
        if 'poro' in features_pred:
            phi_pred = pred_data['poro']
        else:
            phi_pred = phi_exp
        void_ratio_pred = phi_pred / (1.-phi_pred)
        void_ratio_exp = phi_exp / (1.-phi_exp)
        for iload in range(num_loads):
            sig_exp = get_stress_tensor(exp_data, iload)
            sig_pred = get_stress_tensor(pred_data, iload)
            pq_exp[iload, :] = get_p_q(sig_exp)
            pq_pred[iload, :] = get_p_q(sig_pred)
            sig_eigs_exp = np.flip(np.sort(np.linalg.eigvalsh(sig_exp)))
            q123_exp[iload,0] = sig_eigs_exp[0] - sig_eigs_exp[2]
            q123_exp[iload,1] = sig_eigs_exp[0] - sig_eigs_exp[1]
            q123_exp[iload,2] = sig_eigs_exp[1] - sig_eigs_exp[2]
            sig_eigs_pred = np.flip(np.sort(np.linalg.eigvalsh(sig_pred)))
            q123_pred[iload,0] = sig_eigs_pred[0] - sig_eigs_pred[2]
            q123_pred[iload,1] = sig_eigs_pred[0] - sig_eigs_pred[1]
            q123_pred[iload,2] = sig_eigs_pred[1] - sig_eigs_pred[2]
        folder_to_save = folder_add + 'figures/'
        if not os.path.exists(folder_to_save): os.system("mkdir {}".format(folder_to_save))
        plot_x_y([void_ratio_exp, void_ratio_pred],[pq_exp[:,0], pq_pred[:,0]],
                    'void ratio', 'p', ['experiment', 'prediction average'], folder_to_save + 'viodRatio_p.png')
        plot_x_y([eps_11, eps_11],[q123_exp[:,0], q123_pred[:,0]],
                    'axial strain', 'q1=sig1-sig3', ['experiment', 'prediction average'], folder_to_save + 'eps1_q1.png')
        plot_x_y([eps_11, eps_11],[q123_exp[:,1], q123_pred[:,1]],
                    'axial strain', 'q2=sig1-sig2', ['experiment', 'prediction average'], folder_to_save + 'eps1_q2.png')
        plot_x_y([eps_11, eps_11],[q123_exp[:,2], q123_pred[:,2]],
                    'axial strain', 'q3=sig2-sig3', ['experiment', 'prediction average'], folder_to_save + 'eps1_q3.png')

