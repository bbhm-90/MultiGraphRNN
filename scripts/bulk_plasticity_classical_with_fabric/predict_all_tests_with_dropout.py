#####
#    Last update: Oct 1 2020
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
from source.utilities import get_row_ids, prepare_graphs, plot_with_conf_intrv
from source.plotter import plot_x_y

NN_graphs = [
                # # {'name':'0', 'in':["eps11", "eps22","eps33","prev_poro", "init_p"], 'out':['poro']},
                # # {'name':'1', 'in':["eps11", "eps22","eps33", "poro", "prev_sig_eig1", "prev_sig_eig2", "prev_sig_eig3", "init_p"], 'out':["sig_eig1", "sig_eig2", "sig_eig3"]},
                {'name':'0', 'in':["eps11", "eps22","eps33", "init_p"], 'out':['poro']},
                {'name':'1', 'in':["eps11", "eps22","eps33", 'poro', "init_p"], 'out':["sfb11","sfb22","sfb33","sfb12","sfb23","sfb13"]},
                {'name':'2', 'in':["eps11", "eps22","eps33", "poro", "sfb11","sfb22","sfb33","sfb12","sfb23","sfb13", "init_p"], 'out':["sig_eig1", "sig_eig2", "sig_eig3"]},
            ]
dropoutRate = 5
num_mc_trials = 200
root_folder_graph_calc = './results/ANN_graphs_bulk_plasticity_classical_with_fabric/'
num_train_data_sets = num_validation_data_sets = 30
num_points_in_each_data_set = 501
window_size = 40
data_address = "./data/DataHistory_extended_full_tensor_modified2.csv"
num_data_sets_all = 59
data_sets_to_predict = [23, 29, 50, 56]
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
    for ff in out_features:
        temp = data_df_raw[ff].values
        temp = temp[target_ids[0]:target_ids[1]]
        out_data.update({ff:temp})
    return out_data
def add_extra_field_to_dict(out_predic):
    p = np.zeros_like(out_predic['sig_eig1'])
    sig1 = out_predic['sig_eig1']
    sig2 = out_predic['sig_eig2']
    sig3 = out_predic['sig_eig3']
    p = -(sig1+sig2+sig3)/3.
    out_predic.update({'log_p':np.log10(p)})
    out_predic.update({'q1':sig1-sig3})
    out_predic.update({'q2':sig1-sig2})
    out_predic.update({'q3':sig2-sig3})
def add_extra_field_to_df(exp_data):
    p = np.zeros_like(exp_data['sig_eig1'].values)
    sig1 = exp_data['sig_eig1'].values
    sig2 = exp_data['sig_eig2'].values
    sig3 = exp_data['sig_eig3'].values
    p = -(sig1+sig2+sig3)/3.
    exp_data['log_p'] = np.log10(p)
    exp_data['q1'] = sig1-sig3
    exp_data['q2'] = sig1-sig2
    exp_data['q3'] = sig2-sig3

# out_data = data_df_raw[out_features].values
# out_data = out_data[target_ids[0]:target_ids[1], :]

prepare_graphs(NN_graphs, root_dir=root_folder_graph_calc,
               dropout_rates=[dropoutRate/100.,  dropoutRate/100.], num_history=window_size)
root_folder = root_folder_graph_calc + 'data_postproc/'
if not os.path.exists(root_folder): os.system("mkdir {}".format(root_folder))
for data_set_id in data_sets_to_predict:
    exp_data = pd.read_csv(root_folder + 'dataset_{}/data_experiment.csv'.format(data_set_id), delimiter=',')
    in_data = exp_data[in_features].values
    # target_ids  = get_row_ids(data_df_raw, data_set_id, data_set_id+1)
    # in_data = in_data[target_ids[0]:target_ids[1], :]
    assert in_data.shape[0] == num_points_in_each_data_set
    out_predic = complete_prediction(NN_graphs, in_data, num_mc_trials=num_mc_trials, window_size=window_size,
                                    num_points_in_each_data_set=num_points_in_each_data_set,
                                    activated_uncertainity=True)
    # out_exp_data = get_exp_data(out_features, data_df_raw, target_ids)
    temp = root_folder + 'dataset_{}/'.format(data_set_id)
    if not os.path.exists(temp): os.system("mkdir {}".format(temp))
    temp += 'UQ/'
    if not os.path.exists(temp): os.system("mkdir {}".format(temp))
    add_extra_field_to_dict(out_predic)
    add_extra_field_to_df(exp_data)

    if 0:
        max_q = max([np.max(out_predic[i]) for i in ['q1', 'q2', 'q3']])
        min_q = max([np.min(out_predic[i]) for i in ['q1', 'q2', 'q3']])
        range_q = max_q-min_q
    else:
        std_q = np.array([np.std(out_predic[i], axis=1) for i in ['q1', 'q2', 'q3']])
        mean_q = np.array([np.mean(out_predic[i], axis=1) for i in ['q1', 'q2', 'q3']])
        upper_q = np.max(mean_q + 2.*std_q)
        lower_q = np.min(mean_q - 2.*std_q)

    temp_comp_sfb = ['sfb11', 'sfb22', 'sfb33', 'sfb13', 'sfb12', 'sfb23']
    if 0:
        max_fab = max([np.max(out_predic[i]) for i in temp_comp_sfb])
        min_fab = max([np.min(out_predic[i]) for i in temp_comp_sfb])
        range_fab = max_fab-min_fab
    else:
        std_fab = np.array([np.std(out_predic[i], axis=1) for i in temp_comp_sfb])
        mean_fab = np.array([np.mean(out_predic[i], axis=1) for i in temp_comp_sfb])
        upper_fab = np.max(mean_fab + 2.*std_fab)
        lower_fab = np.min(mean_fab - 2.*std_fab)
        

    x_field = -exp_data['eps11'] # make sure compress is positive
    for key, uq_val in out_predic.items():
        if key in {'poro'}:
            vr_uq = out_predic[key] / (1.-out_predic[key])
            vr_exact = exp_data[key].values / (1. - exp_data[key].values)
            plot_with_conf_intrv(vr_uq, vr_exact, x_field, num_mc_trials, num_points_in_each_data_set,
                                x_label='Axial Strain', y_label='Void Ratio', xlim=None, ylim=None,
                                add_to_save=temp+'strain_voidRatio_{}dropRate_numMC{}.png'.format(dropoutRate, num_mc_trials)
                                )
        if key in {'log_p'}:
            plot_with_conf_intrv(out_predic[key], exp_data[key].values, x_field, 
                                 num_mc_trials, num_points_in_each_data_set,
                                x_label='Axial Strain', y_label=r'$\log10(p \mathrm{[KPa]})$', xlim=None, ylim=None,
                                add_to_save=temp+'strain_{}_{}dropRate_numMC{}.png'.format(key, dropoutRate, num_mc_trials)
                                )
        if key in {'q1', 'q2', 'q3'}:
            y_label = None
            if key == 'q1': y_label = r'$q_1 \mathrm{[KPa]}$'
            if key == 'q2': y_label = r'$q_2 \mathrm{[KPa]}$'
            if key == 'q3': y_label = r'$q_3 \mathrm{[KPa]}$'
            plot_with_conf_intrv(out_predic[key], exp_data[key].values, x_field, 
                                 num_mc_trials, num_points_in_each_data_set,
                                x_label='Axial Strain', y_label=y_label,
                                ylim=[lower_q, upper_q],
                                add_to_save=temp+'strain_{}_{}dropRate_numMC{}.png'.format(key, dropoutRate, num_mc_trials)
                                )
        if key in {'sfb11', 'sfb22', 'sfb33', 'sfb13', 'sfb12', 'sfb23'}:
            y_label = None
            if key == 'sfb11': y_label = 'Fabric 11'
            if key == 'sfb22': y_label = 'Fabric 22'
            if key == 'sfb33': y_label = 'Fabric 33'
            if key == 'sfb13': y_label = 'Fabric 13'
            if key == 'sfb12': y_label = 'Fabric 12'
            if key == 'sfb23': y_label = 'Fabric 23'
            plot_with_conf_intrv(out_predic[key], exp_data[key].values, x_field, 
                                 num_mc_trials, num_points_in_each_data_set,
                                x_label='Axial Strain', y_label=y_label,
                                ylim=[lower_fab, upper_fab],
                                add_to_save=temp+'strain_{}_{}dropRate_numMC{}.png'.format(key, dropoutRate, num_mc_trials)
                                )