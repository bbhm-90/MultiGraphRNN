#####
#    Last update: Sep 24 2020
#    Author: Bahador Bahmani bb2969@columbia
#    Under supervision Prof. Waiching Sun
#####
#import joblib
import pandas as pd
import matplotlib.pyplot as plt
from source.preprocessor_ann import reshape_input_data_for_time_history
from source.feed_forward_predictions import complete_prediction
from source.utilities import get_row_ids, prepare_graphs


NN_graphs = [
                # {'name':'0', 'in':["eps11", "eps22","eps33","init_p"], 'out':['degrAssort']},
                # {'name':'1', 'in':["eps11", "eps22","eps33","init_p"], 'out':['average_clustering']},
                # {'name':'2', 'in':["eps11", "eps22","eps33","average_clustering", "init_p"], 'out':['local_efficiency']},
                {'name':'0', 'in':["eps11", "eps22","eps33","init_p"], 'out':['poro']},
                # {'name':'4', 'in':["eps11", "eps22","eps33","poro", "init_p"], 'out':['CN']},
                # {'name':'5', 'in':["eps11", "eps22","eps33","CN", "local_efficiency", "degrAssort", "average_clustering", "init_p"], 'out':['transty']},
                # {'name':'6', 'in':["eps11", "eps22","eps33","init_p"], 'out':['graphDensity']},
                # {'name':'7', 'in':["eps11", "eps22","eps33","poro", "CN", "graphDensity", "transty", "degrAssort", "average_clustering", "local_efficiency", "init_p"], 'out':["sfb11", "sfb22", "sfb33", "sfb12","sfb23", "sfb13"]},
                {'name':'1', 'in':["eps11", "eps22","eps33", "poro", "init_p"], 'out':["sig_eig1", "sig_eig2", "sig_eig3"]},
            ]
root_folder_graph_calc = './results/ANN_graphs_bulk_plasticity_classical_without_fabric/'
num_train_data_sets = num_validation_data_sets = 30
num_points_in_each_data_set = 501
window_size = 10
data_address = "./data/DataHistory_extended_full_tensor_modified2.csv"
num_data_sets_all = 60

data_df_raw = pd.read_csv(data_address, delimiter=',')
# target_ids  = get_row_ids(data_df_raw, 15, 20)
target_ids  = get_row_ids(data_df_raw, 0, 5)
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
out_predic = complete_prediction(NN_graphs, in_data, num_mc_trials=1, window_size=window_size,
                                 num_points_in_each_data_set=num_points_in_each_data_set,
                                 activated_uncertainity=False)
plt.rcParams["figure.figsize"] = (9, 9/1.3)
for key, val in out_predic.items():
    plt.plot(val[:,0], label='prediction', linestyle='--')
    plt.plot(out_data[key], label='experiment')
    plt.ylabel(key, fontsize=22)
    plt.legend(loc="best")
    # plt.savefig('./results/figures/{}.pdf'.format(key))
    plt.show()