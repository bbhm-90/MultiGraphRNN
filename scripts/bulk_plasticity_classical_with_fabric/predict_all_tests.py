#####
#    Last update: Sep 28 2020
#    Author: Bahador Bahmani bb2969@columbia
#    Under supervision Prof. Waiching Sun
#####
#import joblib
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.preprocessor_ann import reshape_input_data_for_time_history
from src.feed_forward_predictions import complete_prediction
from src.utilities import get_row_ids, prepare_graphs


NN_graphs = [
                # # {'name':'0', 'in':["eps11", "eps22","eps33","prev_poro", "init_p"], 'out':['poro']},
                # # {'name':'1', 'in':["eps11", "eps22","eps33", "poro", "prev_sig_eig1", "prev_sig_eig2", "prev_sig_eig3", "init_p"], 'out':["sig_eig1", "sig_eig2", "sig_eig3"]},
                {'name':'0', 'in':["eps11", "eps22","eps33", "init_p"], 'out':['poro']},
                {'name':'1', 'in':["eps11", "eps22","eps33", 'poro', "init_p"], 'out':["sfb11","sfb22","sfb33","sfb12","sfb23","sfb13"]},
                {'name':'2', 'in':["eps11", "eps22","eps33", "poro", "sfb11","sfb22","sfb33","sfb12","sfb23","sfb13", "init_p"], 'out':["sig_eig1", "sig_eig2", "sig_eig3"]},
            ]
num_mc_trials = 1
root_folder_graph_calc = './results/ANN_graphs_bulk_plasticity_classical_with_fabric/'
num_train_data_sets = num_validation_data_sets = 30
num_points_in_each_data_set = 501
window_size = 40
data_address = "./data/DataHistory_extended_full_tensor_modified2.csv"
num_data_sets_all = 59
# # init_set_id=0
# # end_set_id=2

# # data_set_ids = np.array(range(init_set_id, end_set_id))
data_df_raw = pd.read_csv(data_address, delimiter=',')
# target_ids  = get_row_ids(data_df_raw, 15, 20)
in_features = NN_graphs[0]['in'] #["eps11", "eps22","eps33","init_p"]
out_features = list()
for graph in NN_graphs:
    for ff in graph['out']:
        out_features.append(ff)
out_features = list(set(out_features))
# out_features = NN_graphs[-1]['out'] #["eps11", "eps22","eps33","init_p"]

def get_exp_data(out_features, data_df_raw, target_ids):
    out_data = dict()
    out_pred_ave = dict()
    for ff in out_features:
        temp = data_df_raw[ff].values
        temp = temp[target_ids[0]:target_ids[1]]
        out_data.update({ff:temp})
        out_pred_ave.update({ff:np.zeros_like(temp)})
    return out_data, out_pred_ave

# out_data = data_df_raw[out_features].values
# out_data = out_data[target_ids[0]:target_ids[1], :]

prepare_graphs(NN_graphs, root_dir=root_folder_graph_calc)
mse = { i:np.zeros(num_data_sets_all) for i in out_features}
mse.update({'num_loadings':np.zeros(num_data_sets_all, dtype=int) })
root_folder = root_folder_graph_calc + 'data_postproc/'
if not os.path.exists(root_folder): os.system("mkdir {}".format(root_folder))
for data_set_id in range(num_data_sets_all):
    in_data = data_df_raw[in_features].values
    target_ids  = get_row_ids(data_df_raw, data_set_id, data_set_id+1)
    in_data = in_data[target_ids[0]:target_ids[1], :]
    assert in_data.shape[0] == num_points_in_each_data_set
    reshape_input_data_for_time_history(in_data,window_size, num_points_in_each_data_set)
    out_predic = complete_prediction(NN_graphs, in_data, num_mc_trials=num_mc_trials, window_size=window_size,
                                    num_points_in_each_data_set=num_points_in_each_data_set,
                                    activated_uncertainity=False)
    out_data, out_pred_ave = get_exp_data(out_features, data_df_raw, target_ids)
    for key, val in out_predic.items():
        mean_val = np.average(val, axis=1)
        out_pred_ave[key] = mean_val
        mse[key][data_set_id] = np.sum((mean_val-out_data[key])**2)/num_points_in_each_data_set
    mse['num_loadings'][data_set_id] = num_points_in_each_data_set
    temp = root_folder + 'dataset_{}/'.format(data_set_id)
    if not os.path.exists(temp): os.system("mkdir {}".format(temp))
    df  = pd.DataFrame(data_df_raw.values[target_ids[0]:target_ids[1], :],
                        columns=list(data_df_raw.columns))
    df.to_csv(temp+'data_experiment.csv', index=False)
    df = []
    df = pd.DataFrame.from_dict(out_pred_ave)
    df.to_csv(temp+'data_prediction_ave_over_{}.csv'.format(num_mc_trials), index=False)
df = []
df = pd.DataFrame.from_dict(mse)
df.to_csv(root_folder+'mse_over_all_data_sets.csv', index=False)
# plt.rcParams["figure.figsize"] = (9, 9/1.3)
# for key, val in out_predic.items():
#     mean_val = np.average(val, axis=1)
#     plt.plot(mean_val, label='prediction', linestyle='--')
#     plt.plot(out_data[key], label='experiment')
#     plt.ylabel(key, fontsize=22)
#     plt.legend(loc="best")
#     root_folder = './results/figures_ave_{}_to_{}/'.format(init_set_id, end_set_id)
#     if not os.path.exists(root_folder): os.system("mkdir {}".format(root_folder))
#     plt.savefig(root_folder + '{}.pdf'.format(key))
#     # plt.show()
#     plt.close()
    