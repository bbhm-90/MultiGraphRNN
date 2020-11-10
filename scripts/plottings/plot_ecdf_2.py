#####
#    Last update: Sep 28 2020
#    Author: bb2969@columbia
#####
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from source.plotter import plot_x_y
from sklearn.preprocessing import MinMaxScaler
root_folder = './results/'
ann_cases= [
            # 'ANN_graphs_bulk_plasticity_classical_without_fabric_no_do',
            # 'ANN_graphs_bulk_plasticity_classical_without_fabric_yes_do5',
            'ANN_graphs_bulk_plasticity_classical_with_fabric',
            'ANN_graphs_bulk_plasticity_new_fabric_with_classical_graph',
            ]
data_sets_to_analyze = np.array(range(59))
train_data_set_ids = np.array(range(30))
test_data_set_ids = np.array(range(30, 59))
num_loads = 501

ecdf_all_2 = []
y_all = []
for i, case in enumerate(ann_cases):
    ecdf_all = list()
    all_data_exp = pd.read_csv('./data/DataHistory_extended_full_tensor_modified2.csv', delimiter=',')
    folder_add = root_folder + case +'/data_postproc/dataset_{}/'.format(data_sets_to_analyze[0])
    pred_data = pd.read_csv(folder_add + 'data_prediction_ave_over_1.csv', delimiter=',')
    features_pred = list(pred_data.columns)
    all_data_exp = all_data_exp[features_pred].values
    scaler_exp = MinMaxScaler(feature_range=(0, 1))
    scaler_exp.fit(all_data_exp)
    for data_set in data_sets_to_analyze:
        folder_add = root_folder + case +'/data_postproc/dataset_{}/'.format(data_set)
        exp_data = pd.read_csv(folder_add + 'data_experiment.csv', delimiter=',')
        pred_data = pd.read_csv(folder_add + 'data_prediction_ave_over_1.csv', delimiter=',')
        # features_pred = list(pred_data.columns)
        num_rows = exp_data.shape[0]
        assert num_rows == num_loads
        err = np.zeros_like(num_rows)
        scaled_exp_data = exp_data[features_pred].values
        scaled_pred_data = pred_data[features_pred].values
        
        scaled_exp_data = scaler_exp.transform(scaled_exp_data)
        scaled_pred_data = scaler_exp.transform(scaled_pred_data)
        
        err = (scaled_exp_data - scaled_pred_data)**2
        err = np.sum(err, axis=1)
        err /= len(features_pred)
        ecdf_all.append(err)
    
    train_ecdf = []
    for i in train_data_set_ids:
        train_ecdf = train_ecdf + list(ecdf_all[i])
    train_ecdf = np.array(train_ecdf)
    train_ecdf = np.sort(train_ecdf)
    y_train = np.array(range(len(train_ecdf))) / len(train_ecdf)
    
    test_ecdf = []
    for i in test_data_set_ids:
        test_ecdf = test_ecdf + list(ecdf_all[i])
    test_ecdf = np.array(test_ecdf)
    test_ecdf = np.sort(test_ecdf)
    y_test = np.array(range(len(test_ecdf))) / len(test_ecdf)
    ecdf_all_2.append(train_ecdf)
    ecdf_all_2.append(test_ecdf)
    y_all.append(y_train)
    y_all.append(y_test)
    
# plot_x_y(x_all=[train_ecdf, test_ecdf],y_all=[y_train, y_test], x_label='mse', y_label='eCDF', legend_all=['train', 'test'], add_to_save=None)
plot_x_y(x_all=ecdf_all_2,y_all=y_all, x_label='mse', y_label='eCDF',
         legend_all=['train-no graph info', 'test-no graph info', 'train-with graph info', 'test-with graph info'],
         plot_linestyle=['-', '--', '-', '--'],plot_linecolor=['k', 'k', 'b', 'b'],
         xscale='log', add_to_save=None)
exit()