#####
#    Last update: Oct 5 2020
#    Author: Bahador Bahmani bb2969@columbia
#    Under supervision Prof. Waiching Sun
#####
import pandas as pd
import numpy as np
from pickle import load
from src.plotter import plot_x_y
def get_row_ids(datasets, init_id_dataset, end_id_dataset):
    """
        it includes init to end_id_dataset-1!!!
    """
    data_set_ids = datasets['TestCase'].values
    ii = 0
    jj = 0
    res = list()
    for  i, num in enumerate(data_set_ids):
        if num>=init_id_dataset and ii==0:
            res.append(i)
            ii = 1
        if num>=end_id_dataset and jj==0:
            res.append(i)
            jj = 1
        if ii == jj == 1:
            break
    return res

def prepare_graphs(NN_graphs, root_dir, dropout_rates=None, num_history=None):
    from tensorflow.keras.models import load_model
    from src.processor_ann import build_model
    for i, graph in enumerate(NN_graphs):
        name = graph['name']
        scaler_in = load(open(root_dir+name+'/in_data_scaler.pkl', 'rb'))
        # scaler_in = joblib.load(root_dir+name+'/in_data_scaled.pkl')
        scaler_out = load(open(root_dir+name+'/out_data_scaler.pkl', 'rb'))
        if dropout_rates==None:
            model = load_model(root_dir+name+"/trial_model_with_Rdropout_20Percent.h5")
        else:
            temp = pd.read_csv(root_dir+name+'/input_feature_names.csv') 
            numInput = len(temp.columns)
            temp = pd.read_csv(root_dir+name+'/output_feature_names.csv') 
            numOutput = len(temp.columns)
            model = build_model(numInput,numOutput,num_history, dropout_rates)
            model.load_weights(root_dir+name+"/trial_model_with_Rdropout_20Percent.h5")
        NN_graphs[i].update({'input_scaler':scaler_in})
        NN_graphs[i].update({'output_scaler':scaler_out})
        NN_graphs[i].update({'NN_model':model})

def get_stress_tensor(data_df, row_id):
    sig = np.zeros((3,3))
    sig[0,0] = data_df['sig11'].values[row_id]
    sig[1,1] = data_df['sig22'].values[row_id]
    sig[2,2] = data_df['sig33'].values[row_id]
    sig[0,1] = sig[1,0] = data_df['sig12'].values[row_id]
    sig[1,2] = sig[2,1] = data_df['sig23'].values[row_id]
    sig[0,2] = sig[2,0] = data_df['sig13'].values[row_id]
    return sig

def get_sfabric_tensor(data_df, row_id):
    sig = np.zeros((3,3))
    sig[0,0] = data_df['sfb11'].values[row_id]
    sig[1,1] = data_df['sfb22'].values[row_id]
    sig[2,2] = data_df['sfb33'].values[row_id]
    sig[0,1] = sig[1,0] = data_df['sfb12'].values[row_id]
    sig[1,2] = sig[2,1] = data_df['sfb23'].values[row_id]
    sig[0,2] = sig[2,0] = data_df['sfb13'].values[row_id]
    return sig

def get_eigvals(tensor_2nd_order):
    return np.flip(np.sort(np.linalg.eigvalsh(tensor_2nd_order)))


def plot_with_conf_intrv(out_predic, exact_data, x_field, num_mc_trials, num_points_in_each_data_set,
                         x_label, y_label,
                         xlim=None, ylim=None, add_to_save=None):
    assert out_predic.shape[0] == num_points_in_each_data_set
    assert out_predic.shape[1] == num_mc_trials
    assert len(exact_data) == num_points_in_each_data_set
    std = np.std(out_predic, axis=1)
    mean = np.mean(out_predic, axis=1)
    assert len(std) == num_points_in_each_data_set
    assert len(mean) == num_points_in_each_data_set
    plt = plot_x_y(x_all=[x_field, x_field],y_all=[mean, exact_data], x_label=x_label, y_label=y_label,
             legend_all=['Mean Predictions', 'Experiment'],add_leg=False,
             plot_linestyle=['-', '--'], plot_linecolor=['b', 'r'], xlim=xlim, ylim=ylim,
             need_return_plt=True,
             add_to_save=None)
    plt.fill_between(x_field, mean-1.96*std, mean+1.96*std, color='b', alpha=.3) #std curves.
    plt.legend(loc="best", prop={'size': 28})
    plt.tight_layout()
    if add_to_save:
        plt.savefig(add_to_save)
        plt.close()
    else:
        plt.show()

def get_angle_two_unit_vectors(v1, v2):
    assert np.allclose(np.dot(v1, v1), 1.)
    assert np.allclose(np.dot(v2, v2), 1.)
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))*180./np.pi