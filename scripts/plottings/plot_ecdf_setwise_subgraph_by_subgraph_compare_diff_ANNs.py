#####
#    Last update: Sep 28 2020
#    Author: bb2969@columbia
#####
"""
    in final curve:
        each point represents one dataset
    for each subgraph we have a different performance curve
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.plotter import plot_x_y
from sklearn.preprocessing import MinMaxScaler
root_folder = './results/'
ann_cases= [
            {'main_graph_name':'ANN_graphs_bulk_plasticity_classical_with_fabric',
                'sub_graph_outputs':[['poro'], ["sfb11","sfb22","sfb33","sfb12","sfb23","sfb13"],
                                        ["sig_eig1", "sig_eig2", "sig_eig3"]],
                'output_name':['porosity', 'fabric', 'stress_eigenvalues'],},
            {'main_graph_name':'ANN_graphs_bulk_plasticity_new_fabric_with_classical_graph',
                'sub_graph_outputs':[['poro'], ['degrAssort'],['average_clustering'],
                                     ['graphDensity'],['local_efficiency'],['CN'],['transty'],
                                     ["sfb11","sfb22","sfb33","sfb12","sfb23","sfb13"],
                                     ["sig_eig1", "sig_eig2", "sig_eig3"]],
                'output_name':['porosity', 'degrAssort', 'average_clustering', 'graphDensity',
                                'local_efficiency', 'CN', 'transty',
                                'fabric', 'stress_eigenvalues'],},
            ]
target_fields = ['porosity', 'fabric', 'stress_eigenvalues']
data_sets_to_analyze = np.array(range(59))
train_data_set_ids = np.array(range(30))
test_data_set_ids = np.array(range(30, 59))
num_loads = 501


for field in target_fields:
    train_ecdf_all_ann_cases=[]
    y_train_all_ann_cases=[]
    test_ecdf_all_ann_cases=[]
    y_test_all_ann_cases=[]
    for i, case in enumerate(ann_cases):    
        folder_add = root_folder + case['main_graph_name'] +'/data_postproc/'
        temp = np.load(folder_add + 'train_setwise_ecdf_{}.npy'.format(field))
        train_ecdf, y_train = temp[:,0], temp[:,1]
        train_ecdf_all_ann_cases.append(train_ecdf)
        y_train_all_ann_cases.append(y_train)
        temp = np.load(folder_add + 'test_setwise_ecdf_{}.npy'.format(field))
        test_ecdf, y_test = temp[:,0], temp[:,1]
        test_ecdf_all_ann_cases.append(test_ecdf)
        y_test_all_ann_cases.append(y_test)
    
    plot_x_y(x_all=[train_ecdf_all_ann_cases[0], test_ecdf_all_ann_cases[0], train_ecdf_all_ann_cases[1], test_ecdf_all_ann_cases[1]],
            y_all=[y_train_all_ann_cases[0], y_test_all_ann_cases[0], y_train_all_ann_cases[1], y_test_all_ann_cases[1]],
             x_label='Scaled MSE', y_label='eCDF',
             legend_all=['train-no graph info', 'test-no graph info', 'train-with graph info', 'test-with graph info'],
             plot_linestyle=['-', '--', '-', '--'],plot_linecolor=['k', 'k', 'b', 'b'],
             xscale='log',
             add_to_save=folder_add + 'compare_setwise_ecdf_{}.png'.format(field),
             )