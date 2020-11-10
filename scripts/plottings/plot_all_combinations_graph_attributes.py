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
            'ANN_graphs_bulk_plasticity_new_fabric_with_classical_graph',
            ]
target_fields = ["degrAssort", "average_clustering", "graphDensity", "local_efficiency","CN", "transty"]
comp_mapped = ['Degree Assortativity', 'Average Clustering', 'Graph Density', 'Local Efficiency', 'Coordination Number', 'Graph Transitivity ']
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
        eps_11 = -exp_data['eps11'].values

        fabric_exp = exp_data[target_fields].values
        fabric_pred = pred_data[target_fields].values
        
        folder_to_save = folder_add + 'figures/'
        if not os.path.exists(folder_to_save): os.system("mkdir {}".format(folder_to_save))

        for col_id in range(fabric_exp.shape[1]):
            y_exp = fabric_exp[:, col_id]
            y_pred = fabric_pred[:, col_id]
            plot_x_y([eps_11, eps_11],[y_exp, y_pred],
                        'Axial Strain', '{}'.format(comp_mapped[col_id]), ['Experiment', 'Prediction'],
                        add_to_save=folder_to_save + 'eps1_{}.png'.format(target_fields[col_id]),
                        )

