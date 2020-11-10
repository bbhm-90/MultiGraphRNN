#####
#    Last update: Sep 28 2020
#    Author: bb2969@columbia
#####
import os
import pandas as pd
import numpy as np
from src.plotter import plot_x_y
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
        folder_to_save = folder_add + 'figures/'
        if not os.path.exists(folder_to_save): os.system("mkdir {}".format(folder_to_save))
        for feature_name in features_pred:
            plot_x_y([range(num_loads), range(num_loads)], [exp_data[feature_name], pred_data[feature_name]],
                        'load step', feature_name, ['experiment', 'prediction'], folder_to_save + 'loadStep_{}.png'.format(feature_name))


