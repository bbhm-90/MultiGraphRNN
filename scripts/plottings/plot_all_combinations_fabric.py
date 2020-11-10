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
target_fields = ["sfb11", "sfb22", "sfb33", "sfb12","sfb23", "sfb13"]
comp_mapped = ['11', '22', '33', '12', '23', '13']
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

        temp_max = np.max(fabric_exp, axis=0)
        assert len(temp_max) == len(target_fields)
        temp_max = np.max(temp_max)

        temp_min = np.min(fabric_exp, axis=0)
        assert len(temp_min) == len(target_fields)
        temp_min = np.min(temp_min)
        
        range_val = abs(temp_max - temp_min)
        for col_id in range(fabric_exp.shape[1]):
            y_exp = fabric_exp[:, col_id]
            y_pred = fabric_pred[:, col_id]
            plot_x_y([eps_11, eps_11],[y_exp, y_pred],
                        'Axial Strain', 'Fabric {}'.format(comp_mapped[col_id]), ['Experiment', 'Prediction'],
                        ylim=[temp_min-0.05*range_val, temp_max+0.05*range_val],
                        add_to_save=folder_to_save + 'eps1_fab{}.png'.format(comp_mapped[col_id]),
                        )

