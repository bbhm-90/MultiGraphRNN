#####
#    Last update: Sep 28 2020
#    Author: Bahador Bahmani bb2969@columbia
#    Under supervision Prof. Waiching Sun
#####
import pandas as pd
import numpy as np
from src.preprocessor_ann import creat_input_output_files
from src.processor_ann import train_ann
from src.utilities import get_eigvals
def add_modification_culumns(data_raw_df, num_data_sets, num_points_in_each_data_set):
    assert data_raw_df.shape[0] == num_data_sets*num_points_in_each_data_set
    transformed_p = []
    for i in range(num_data_sets):
        initial_p = data_raw_df.p.iloc[i*(num_points_in_each_data_set-1) + i]
        transformed_p.extend([initial_p for i in range(num_points_in_each_data_set)])
    data_raw_df = data_raw_df.fillna(0)
    data_raw_df["init_p"] = transformed_p
    data_raw_df["isInit"] = 0
    for i in range(num_data_sets):
        data_raw_df.isInit.iloc[i*(num_points_in_each_data_set-1) + i] = 1
    return data_raw_df

def add_stress_eigvals(data_df):
    eigvals = np.zeros((data_df.shape[0],3))
    for index, row in data_df.iterrows():
        sig = np.zeros((3,3))
        sig[0,0] = row['sig11']
        sig[1,1] = row['sig22']
        sig[2,2] = row['sig33']
        sig[0,1] = sig[1,0] = row['sig12']
        sig[1,2] = sig[2,1] = row['sig23']
        sig[0,2] = sig[2,0] = row['sig13']
        eigvals[index,:] = get_eigvals(sig)
    data_df['sig_eig1'] = eigvals[:,0]
    data_df['sig_eig2'] = eigvals[:,1]
    data_df['sig_eig3'] = eigvals[:,2]
    return data_df

def add_prev_step(data_df, num_points_in_each_data_set):
    df_col_names = list(data_df.columns)
    num_rows = data_df.shape[0]
    assert num_rows%num_points_in_each_data_set==0
    num_data_sets = num_rows//num_points_in_each_data_set
    temp = [0]
    while len(temp) < num_data_sets:
        temp.append(num_points_in_each_data_set+temp[-1])
    for col in df_col_names:
        data_df['prev_' + col] = np.zeros(num_rows)
        data_df.loc[1:, 'prev_' + col] = data_df.loc[0:num_rows, col]
        data_df.loc[temp, 'prev_' + col] = 0.
    return data_df

if __name__ == "__main__":
    root_folder_graph_calc = './results/ANN_graphs_bulk_plasticity_classical_without_fabric/'
    num_train_data_sets = num_validation_data_sets = 30
    num_points_in_each_data_set = 501
    window_size = 40
    data_address = "./data/DataHistory_extended_full_tensor.csv"
    num_data_sets_all = 60
    data_df_raw = pd.read_csv(data_address, delimiter=',')
    assert data_df_raw.shape[0] == num_data_sets_all*num_points_in_each_data_set
    data_df_raw = add_modification_culumns(data_df_raw,num_data_sets_all,num_points_in_each_data_set)
    data_df_raw = add_stress_eigvals(data_df_raw)
    data_df_raw = add_prev_step(data_df_raw, num_points_in_each_data_set)
    data_df_raw.to_csv("./data/DataHistory_extended_full_tensor_modified2.csv", index=False)
    # data_df_raw = pd.read_csv(data_address, delimiter=',')
    print("DONE!")
