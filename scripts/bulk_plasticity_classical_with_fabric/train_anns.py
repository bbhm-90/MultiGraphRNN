#####
#    Last update: Sep 28 2020
#    Author: Bahador Bahmani bb2969@columbia
#    Under supervision Prof. Waiching Sun
#####
import pandas as pd
from source.preprocessor_ann import creat_input_output_files
from source.processor_ann import train_ann
NN_graphs = [
                # # {'name':'0', 'in':["eps11", "eps22","eps33","prev_poro", "init_p"], 'out':['poro']},
                # # {'name':'1', 'in':["eps11", "eps22","eps33", "poro", "prev_sig_eig1", "prev_sig_eig2", "prev_sig_eig3", "init_p"], 'out':["sig_eig1", "sig_eig2", "sig_eig3"]},
                {'name':'0', 'in':["eps11", "eps22","eps33", "init_p"], 'out':['poro']},
                {'name':'1', 'in':["eps11", "eps22","eps33", 'poro', "init_p"], 'out':["sfb11","sfb22","sfb33","sfb12","sfb23","sfb13"]},
                {'name':'2', 'in':["eps11", "eps22","eps33", "poro", "sfb11","sfb22","sfb33","sfb12","sfb23","sfb13", "init_p"], 'out':["sig_eig1", "sig_eig2", "sig_eig3"]},
            ]
root_folder_graph_calc = './results/ANN_graphs_bulk_plasticity_classical_with_fabric/'
num_train_data_sets = num_validation_data_sets = 30
num_points_in_each_data_set = 501
window_size = 40
data_address = "./data/DataHistory_extended_full_tensor_modified2.csv"
num_data_sets_all = 60
data_df_raw = pd.read_csv(data_address, delimiter=',')

for i, graph in enumerate(NN_graphs):
    graph_name = graph['name']
    creat_input_output_files(data_df=data_df_raw,num_train_data_sets=num_train_data_sets,
                            num_validation_data_sets=num_validation_data_sets,
                            window_size=window_size, num_points_in_each_data_set=num_points_in_each_data_set,
                            graph=graph,graph_name=graph_name, root_folder=root_folder_graph_calc)

for graph in NN_graphs:
    ann_graph_folder = graph['name']
    train_ann(root_folder_graph_calc + ann_graph_folder + '/')