#####
#    Last update: Sep 28 2020
#    Author: bb2969@columbia
#####
"""
    in final curve:
        each point represents one dataset
    one plot for the entire of prediction flow
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from source.plotter import plot_x_y
from sklearn.preprocessing import MinMaxScaler
root_folder = './results/'
ann_cases= [
            # 'ANN_graphs_bulk_plasticity_classical_without_fabric',
            {'main_graph_name':'ANN_graphs_bulk_plasticity_classical_with_fabric',
                'sub_graph_outputs':[['poro'], ["sfb11","sfb22","sfb33","sfb12","sfb23","sfb13"],
                                        ["sig_eig1", "sig_eig2", "sig_eig3"]],
                'output_name':['porosity', 'fabric', 'stress_eigenvalues'],}
            # 'ANN_graphs_bulk_plasticity_new_fabric_with_classical_graph',
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
    error_sets = np.zeros((num_datasets))
    # for sub_graph_output in sub_graph_outputs:
    #     num_comp = len(sub_graph_output)
    #     for data_set in data_sets_to_analyze:

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
        num_elem = err.shape[0] * err.shape[1]
        error_sets[j] = np.sum(err) / num_elem
    temp = root_folder + case['main_graph_name'] +'/data_postproc/'
    train_ecdf = error_sets[train_data_set_ids]
    train_ecdf = np.array(train_ecdf)
    train_ecdf = np.sort(train_ecdf)
    y_train = np.array(range(len(train_ecdf))) / len(train_ecdf)

    test_ecdf = error_sets[test_data_set_ids]
    test_ecdf = np.array(test_ecdf)
    test_ecdf = np.sort(test_ecdf)
    y_test = np.array(range(len(test_ecdf))) / len(test_ecdf)
    plot_x_y(x_all=[train_ecdf, test_ecdf],y_all=[y_train, y_test], x_label='scaled MSE', y_label='eCDF',
                legend_all=['Train', 'Test'],xscale='log',
                add_to_save=temp+'setwise_ecdf_all.png'
                )
    