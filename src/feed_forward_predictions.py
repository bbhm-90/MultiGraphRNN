#####
#    Last update: Sep 28 2020
#    Author: Bahador Bahmani bb2969@columbia
#    Under supervision Prof. Waiching Sun
#####
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from source.preprocessor_ann import reshape_input_data_for_time_history

def complete_prediction(NN_graphs, x_in, num_mc_trials, window_size, num_points_in_each_data_set,
                            activated_uncertainity=True):
    """
        x_in:must be in real scale
        NN_graphs = [  
                    {'name':'0', 'in':["eps11", "eps22","eps33","init_p"], 'out':['epsv'],
                    'NN_model':model0, 'input_scaler':input_scaler0, 'output_scaler':output_scaler0},
                    {'name':'1', 'in':["eps11", "eps22","eps33","epsv", "init_p"], 'out':["fb11","fb22","fb33","fb12","fb23","fb13"],
                    'NN_model':model0, 'input_scaler':input_scaler0, 'output_scaler':output_scaler0},
                    ]        
    """
    if not activated_uncertainity: assert num_mc_trials==1
    num_load_steps = x_in.shape[0]
    num_input_feature_base = x_in.shape[1]
    input_feature_base = NN_graphs[0]['in']
    assert num_input_feature_base == len(input_feature_base)
    map_feature_base = {name:i for i, name in enumerate(input_feature_base)}
    all_prediction_featues = list()
    for i in NN_graphs:
        for j in i['out']:
            all_prediction_featues.append(j)
    all_prediction_featues = list(set(all_prediction_featues))
    all_predictions = dict() # in real scale and without time seri format
    for i in range(num_mc_trials):
        for graph in NN_graphs:
            current_in_features = graph['in']
            x_raw = np.zeros(shape=(num_load_steps, len(current_in_features)))
            for j, feature_name in enumerate(current_in_features):
                if feature_name in all_prediction_featues:
                    x_raw[:, j] = all_predictions[feature_name][:, i]
                else:
                    x_raw[:, j] = x_in[:, map_feature_base[feature_name]]
            scalar_in = graph['input_scaler']
            x = scalar_in.transform(x_raw)
            x = reshape_input_data_for_time_history(x, window_size, num_points_in_each_data_set)
            if activated_uncertainity:
                pred = graph['NN_model'](x,training = True,).numpy() # for dropout activation
            else:
                pred = graph['NN_model'].predict(x) # for deterministic
            scalar_out = graph['output_scaler']
            pred = scalar_out.inverse_transform(pred)
            current_out_features = graph['out']
            for j, feature_name in enumerate(current_out_features):
                if not (feature_name in all_predictions.keys()):
                    all_predictions.update({feature_name:np.zeros(shape=(num_load_steps, num_mc_trials))})
                all_predictions[feature_name][:, i] = pred[:, j]
    
    return all_predictions


