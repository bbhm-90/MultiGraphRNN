#####
#    Last update: Sep 24 2020
#    Author: Bahador Bahmani bb2969@columbia
#    Under supervision Prof. Waiching Sun
#####
import os
import numpy as np
import pandas as pd
from pickle import dump, load
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

def reshape_input_data_for_time_history(all_data_sets_in, window_size, num_points_in_each_data_set):
    num_total_points = len(all_data_sets_in[:,0])
    num_features = len(all_data_sets_in[0,:])
    assert num_total_points%num_points_in_each_data_set == 0
    num_data_sets = num_total_points//num_points_in_each_data_set
    X = np.zeros((num_total_points,window_size,num_features), dtype=float)
    count =0
    for i_data_set in range(num_data_sets):
        temp = np.zeros(shape=(num_points_in_each_data_set+window_size-1, num_features), dtype=float) # paded zeros for first load steps
        start_id = i_data_set*num_points_in_each_data_set
        temp[window_size-1:, :] = all_data_sets_in[start_id:start_id+num_points_in_each_data_set, :]
        for i in range(num_points_in_each_data_set):
            X[count,:,:] = temp[i:i+window_size,:] # each load step has a history with size windowsize
            count += 1
    return X

def scale_data(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data), scaler

def creat_input_output_files(data_df, num_train_data_sets, num_validation_data_sets,
                    window_size, num_points_in_each_data_set, graph, graph_name, root_folder):
    """
        Ref:
            some parts grabbed from Dr. Kun Wang code
    """
    assert data_df.shape[0] % num_points_in_each_data_set == 0
    graph_folder = root_folder+graph_name+'/'
    in_features = graph['in']
    out_features = graph['out']
    in_data, out_data = data_df[in_features], data_df[out_features]
    in_data, out_data = in_data.values, out_data.values

    if not os.path.exists(root_folder): os.system("mkdir {}".format(root_folder))
    if not os.path.exists(graph_folder): os.system("mkdir {}".format(graph_folder))
    in_data_scaled, in_data_scaler = scale_data(in_data)
    joblib.dump(in_data_scaled, graph_folder+'in_data_scaled.pkl')
    dump(in_data_scaler, open(graph_folder+'in_data_scaler.pkl', 'wb'))
    # joblib.dump(in_data_scalar, graph_folder+'in_data_scaler.pkl')
    pd.DataFrame(data={i :[] for i in in_features}).to_csv(graph_folder+'input_feature_names.csv', index=False)

    out_data_scaled, out_data_scaler = scale_data(out_data)
    joblib.dump(out_data_scaled, graph_folder+'out_data_scaled.pkl')
    dump(out_data_scaler, open(graph_folder+'out_data_scaler.pkl', 'wb'))
    # print(load(open(graph_folder+'out_data_scaler.pkl', 'rb')))
    # exit()
    pd.DataFrame(data={i :[] for i in out_features}).to_csv(graph_folder+'output_feature_names.csv', index=False)

    traindata_index = len(data_df.loc[data_df['TestCase'] < num_train_data_sets])
    validdata_index = len(data_df.loc[data_df['TestCase'] < num_train_data_sets+num_validation_data_sets])
    in_data_scaled = reshape_input_data_for_time_history(in_data_scaled, window_size,num_points_in_each_data_set)

    np.save(graph_folder+'Training_Data_Input.npy', in_data_scaled[0:traindata_index,:,:])
    np.save(graph_folder+'Validation_Data_Input.npy', in_data_scaled[traindata_index:validdata_index,:,:])
    np.save(graph_folder+'Training_Data_Output.npy', out_data_scaled[0:traindata_index,:])
    np.save(graph_folder+'Validation_Data_Output.npy', out_data_scaled[traindata_index:validdata_index,:])

if __name__ == "__main__":
    NN_graphs = [{'in':["eps11", "eps22","eps33","init_p"], 'out':['epsv']},
                {'in':["eps11", "eps22","eps33","epsv", "init_p"], 'out':["fb11","fb22","fb33","fb12","fb23","fb13"]},
                {'in':["eps11", "eps22","eps33","fb11","fb22","fb33","fb12","fb23","fb13", "init_p"], 'out':["p","q"]},
                ]
    data_address = "./DataHistory_extended_modified.csv"
    data_df = pd.read_csv(data_address, delimiter=',')
    num_train_data_sets = num_validation_data_sets = 30
    num_points_in_each_data_set = 501
    window_size = 20
    graph = NN_graphs[0]
    creat_input_output_files(data_df=data_df,num_train_data_sets=num_train_data_sets, 
                             num_validation_data_sets=num_validation_data_sets,
                             window_size=window_size, num_points_in_each_data_set=num_points_in_each_data_set,
                             graph=graph,graph_name="0")