#####
#    Last update: Sep 24 2020
#    Author: Bahador Bahmani bb2969@columbia
#    Under supervision Prof. Waiching Sun
#####
#import joblib
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from source.preprocessor_ann import reshape_input_data_for_time_history
from source.feed_forward_predictions import complete_prediction
from source.utilities import get_row_ids, prepare_graphs


NN_graphs = [
                {'name':'0', 'in':["eps11", "eps22","eps33","init_p"], 'out':['degrAssort']},
                {'name':'1', 'in':["eps11", "eps22","eps33","init_p"], 'out':['average_clustering']},
                {'name':'2', 'in':["eps11", "eps22","eps33","average_clustering", "init_p"], 'out':['local_efficiency']},
                {'name':'3', 'in':["eps11", "eps22","eps33","init_p"], 'out':['poro']},
                {'name':'4', 'in':["eps11", "eps22","eps33","poro", "init_p"], 'out':['CN']},
                {'name':'5', 'in':["eps11", "eps22","eps33","CN", "local_efficiency", "degrAssort", "average_clustering", "init_p"], 'out':['transty']},
                {'name':'6', 'in':["eps11", "eps22","eps33","init_p"], 'out':['graphDensity']},
                {'name':'7', 'in':["eps11", "eps22","eps33","poro", "CN", "graphDensity", "transty", "degrAssort", "average_clustering", "local_efficiency", "init_p"], 'out':["sfb11", "sfb22", "sfb33", "sfb12","sfb23", "sfb13"]},
                # {'name':'8', 'in':["sfb11","sfb22","sfb33","sfb12","sfb23","sfb13", "init_p"], 'out':["sig11", "sig22", "sig33", "sig12", "sig23", "sig13"]},
                {'name':'8', 'in':["eps11", "eps22","eps33", "poro", "sfb11","sfb22","sfb33","sfb12","sfb23","sfb13", "init_p"], 'out':["sig11", "sig22", "sig33", "sig12", "sig23", "sig13"]},
            ]
root_folder_graph_calc = './results/ANN_graphs_bulk_plasticity_new_fabric_with_classical_graph/'
num_train_data_sets = num_validation_data_sets = 30
num_points_in_each_data_set = 501
window_size = 20
data_address = "./data/DataHistory_extended_full_tensor_modified.csv"
num_data_sets_all = 60
init_set_id=int(sys.argv[1])
end_set_id=int(sys.argv[2])

data_df_raw = pd.read_csv(data_address, delimiter=',')
# target_ids  = get_row_ids(data_df_raw, 15, 20)
target_ids  = get_row_ids(data_df_raw, init_set_id, end_set_id)
in_features = NN_graphs[0]['in'] #["eps11", "eps22","eps33","init_p"]
in_data = data_df_raw[in_features].values
in_data = in_data[target_ids[0]:target_ids[1], :]
out_features = list()
for graph in NN_graphs:
    for ff in graph['out']:
        out_features.append(ff)
out_features = list(set(out_features))
# out_features = NN_graphs[-1]['out'] #["eps11", "eps22","eps33","init_p"]
out_data = dict()
for ff in out_features:
    temp = data_df_raw[ff].values
    temp = temp[target_ids[0]:target_ids[1]]
    out_data.update({ff:temp})

# out_data = data_df_raw[out_features].values
# out_data = out_data[target_ids[0]:target_ids[1], :]
reshape_input_data_for_time_history(in_data,window_size, num_points_in_each_data_set)

prepare_graphs(NN_graphs, root_dir=root_folder_graph_calc)
out_predic = complete_prediction(NN_graphs, in_data, num_mc_trials=200, window_size=window_size,
                                 num_points_in_each_data_set=num_points_in_each_data_set)
plt.rcParams["figure.figsize"] = (9, 9/1.3)
for key, val in out_predic.items():
    mean_val = np.average(val, axis=1)
    plt.plot(mean_val, label='prediction', linestyle='--')
    plt.plot(out_data[key], label='experiment')
    plt.ylabel(key, fontsize=22)
    plt.legend(loc="best")
    root_folder = './results/figures_ANN_graphs_bulk_plasticity_new_fabric_with_classical_graph/ave_{}_to_{}/'.format(init_set_id, end_set_id)
    if not os.path.exists(root_folder): os.system("mkdir {}".format(root_folder))
    plt.savefig(root_folder + '{}.pdf'.format(key))
    # plt.show()
    plt.close()
    