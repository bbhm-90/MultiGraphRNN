#####
#    Last update: Sep 28 2020
#    Author: bb2969@columbia
#####
import os
import pandas as pd
import numpy as np
from source.plotter import plot_x_y
root_folder = './results/'
ann_cases= [
            # 'ANN_graphs_bulk_plasticity_classical_without_fabric',
            'ANN_graphs_bulk_plasticity_classical_with_fabric',
            # 'ANN_graphs_bulk_plasticity_new_fabric_with_classical_graph',
            ]
data_sets_to_analyze = np.array(range(59))
num_loads = 501


q123_exp = np.zeros((num_loads, 3))
q123_pred = np.zeros((num_loads, 3))
pq_exp = np.zeros((num_loads,2))
pq_pred = np.zeros((num_loads,2))
for i, case in enumerate(ann_cases):
    for data_set in data_sets_to_analyze:
        folder_add = root_folder + case +'/data_postproc/dataset_{}/'.format(data_set)
        exp_data = pd.read_csv(folder_add + 'data_experiment.csv', delimiter=',')
        pred_data = pd.read_csv(folder_add + 'data_prediction_ave_over_1.csv', delimiter=',')
        features_pred = list(pred_data.columns)
        eps_11 = -exp_data['eps11'].values
        phi_exp = exp_data['poro'].values
        if 'poro' in features_pred:
            phi_pred = pred_data['poro']
        else:
            phi_pred = phi_exp
        void_ratio_pred = phi_pred / (1.-phi_pred)
        void_ratio_exp = phi_exp / (1.-phi_exp)
        
        sig_eig1_exp = exp_data['sig_eig1'].values
        sig_eig2_exp = exp_data['sig_eig2'].values
        sig_eig3_exp = exp_data['sig_eig3'].values
        
        sig_eig1_pred = pred_data['sig_eig1'].values
        sig_eig2_pred = pred_data['sig_eig2'].values
        sig_eig3_pred = pred_data['sig_eig3'].values
        
        p_exp = -(sig_eig1_exp + sig_eig2_exp + sig_eig3_exp) / 3.
        q1_exp = sig_eig1_exp - sig_eig3_exp
        q2_exp = sig_eig1_exp - sig_eig2_exp
        q3_exp = sig_eig2_exp - sig_eig3_exp
        
        p_pred = -(sig_eig1_pred + sig_eig2_pred + sig_eig3_pred) / 3.
        q1_pred = sig_eig1_pred- sig_eig3_pred
        q2_pred = sig_eig1_pred - sig_eig2_pred
        q3_pred = sig_eig2_pred - sig_eig3_pred
        
        folder_to_save = folder_add + 'figures/'
        if not os.path.exists(folder_to_save): os.system("mkdir {}".format(folder_to_save))
        plot_x_y([np.log10(p_exp), np.log10(p_pred)], [void_ratio_exp, void_ratio_pred],
                    r'$\log10(p [\mathrm{KPa}] )$', 'Void Ratio', ['Experiment', 'Prediction'],
                    folder_to_save + 'log10p_voidRatio.png'
                    )
        max_y = max(max(q1_exp), max(q2_exp), max(q3_exp))
        min_y = min(min(q1_exp), min(q2_exp), min(q3_exp))
        range_y = max_y-min_y
        plot_x_y([eps_11, eps_11],[q1_exp, q1_pred],
                    'Axial Strain', r'$q_1 [\mathrm{KPa}]$', ['Experiment', 'Prediction'],
                    ylim=[min_y-0.05*range_y, max_y+0.05*range_y],
                    add_to_save=folder_to_save + 'eps1_q1.png'
                    )

        plot_x_y([eps_11, eps_11],[q2_exp, q2_pred],
                    'Axial Strain', r'$q_2 [\mathrm{KPa}]$', ['Experiment', 'Prediction'],
                    ylim=[min_y-0.05*range_y, max_y+0.05*range_y],
                    add_to_save=folder_to_save + 'eps1_q2.png'
                    )
        plot_x_y([eps_11, eps_11],[q3_exp, q3_pred],
                    'Axial Strain', r'$q_3 [\mathrm{KPa}]$', ['Experiment', 'Prediction'],
                    ylim=[min_y-0.05*range_y, max_y+0.05*range_y],
                    add_to_save=folder_to_save + 'eps1_q3.png'
                    )

