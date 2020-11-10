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
            # {'main_graph_name':'ANN_graphs_bulk_plasticity_new_fabric_with_classical_graph',
            #     'sub_graph_outputs':[['poro'], ['degrAssort'],['average_clustering'],
            #                          ['graphDensity'],['local_efficiency'],['CN'],['transty'],
            #                          ["sfb11","sfb22","sfb33","sfb12","sfb23","sfb13"],
            #                          ["sig_eig1", "sig_eig2", "sig_eig3"]],
            #     'output_name':['porosity', 'degrAssort', 'average_clustering', 'graphDensity',
            #                     'local_efficiency', 'CN', 'transty',
            #                     'fabric', 'stress_eigenvalues'],},
            ]

data_sets_to_analyze = np.array(range(59))
train_data_set_ids = np.array(range(30))
test_data_set_ids = np.array(range(30, 59))
num_loads = 501


for i, case in enumerate(ann_cases):
    ecdf_all = list()
    all_data_exp = pd.read_csv('./data/DataHistory_extended_full_tensor_modified2.csv', delimiter=',')
    folder_add = root_folder + case['main_graph_name'] +'/data_postproc/dataset_{}/'.format(data_sets_to_analyze[0])
    pred_data = pd.read_csv(folder_add + 'data_prediction_ave_over_1.csv', delimiter=',')
    features_pred = list(pred_data.columns)
    features_pred_map = {ff:i for i, ff in enumerate(features_pred)}
    all_data_exp = all_data_exp[features_pred].values
    scaler_exp = MinMaxScaler(feature_range=(0, 1))
    scaler_exp.fit(all_data_exp)
    sub_graph_outputs = case['sub_graph_outputs']
    num_anns = len(sub_graph_outputs)
    num_out_feature_each_ann = np.array([len(gg) for gg in sub_graph_outputs], dtype=int)
    num_datasets = len(data_sets_to_analyze)
    error_anns = np.zeros((num_datasets, num_anns))
    # for sub_graph_output in sub_graph_outputs:
    #     num_comp = len(sub_graph_output)
    #     for data_set in data_sets_to_analyze:
    all_errors = []
    for j, data_set in enumerate(data_sets_to_analyze):
        folder_add = root_folder + case['main_graph_name'] +'/data_postproc/dataset_{}/'.format(data_set)
        exp_data = pd.read_csv(folder_add + 'data_experiment.csv', delimiter=',')
        pred_data = pd.read_csv(folder_add + 'data_prediction_ave_over_1.csv', delimiter=',')
        # features_pred = list(pred_data.columns)
        num_rows = exp_data.shape[0]
        assert num_rows == num_loads
        scaled_exp_data = exp_data[features_pred].values
        scaled_pred_data = pred_data[features_pred].values
        
        scaled_exp_data = scaler_exp.transform(scaled_exp_data)
        scaled_pred_data = scaler_exp.transform(scaled_pred_data)
        
        err = (scaled_exp_data - scaled_pred_data)**2
        err = np.sum(err, axis=1)
        assert len(err) == num_loads
        all_errors.append(err)
    temp_root_folder = root_folder + case['main_graph_name'] +'/data_postproc/'
    train_ecdf = []
    for train_id in train_data_set_ids:
        temp = all_errors[train_id]
        train_ecdf = train_ecdf + list(temp)
    train_ecdf = np.array(train_ecdf)
    train_ecdf = np.sort(train_ecdf)
    if len(train_ecdf)>1: assert train_ecdf[0]<= train_ecdf[1]
    y_train = np.array(range(len(train_ecdf))) / len(train_ecdf)

    test_ecdf = []
    for test_id in test_data_set_ids:
        temp = all_errors[test_id]
        test_ecdf = test_ecdf + list(temp)
    test_ecdf = np.array(test_ecdf)
    test_ecdf = np.sort(test_ecdf)
    if len(test_ecdf)>1: assert test_ecdf[0]<= test_ecdf[1]
    y_test = np.array(range(len(test_ecdf))) / len(test_ecdf)
    
    temp = np.zeros((len(train_ecdf), 2))
    temp[:,0], temp[:,1] = train_ecdf, y_train
    np.save(temp_root_folder+'train_pointwise_ecdf_all.npy', temp)
    temp = np.zeros((len(test_ecdf), 2))
    temp[:,0], temp[:,1] = test_ecdf, y_test
    np.save(temp_root_folder+'test_pointwise_ecdf_all.npy', temp)

    plot_x_y(x_all=[train_ecdf, test_ecdf],y_all=[y_train, y_test], x_label='scaled MSE', y_label='eCDF',
                legend_all=['train', 'test'], xscale='log',
                add_to_save=temp_root_folder + 'pointwise_all.png',
                )
    