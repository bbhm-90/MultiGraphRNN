#####
#    Last update: Oct 5 2020
#    Author: Bahador Bahmani bb2969@columbia
#    Under supervision Prof. Waiching Sun
#####
import numpy as np
import pandas as pd
from source.utilities import get_stress_tensor, get_sfabric_tensor, plot_x_y
datset_ids = range(59)
b_values = [0.]*6+[0.5]*6+[0.1]*6+[0.25]*6+[0.75]*6
b_values = b_values + b_values
# datset_ids = [0, 23, 29, 50, 56]
num_load_steps = 501
root_folder_graph_calc = './results/ANN_graphs_bulk_plasticity_classical_with_fabric/'

def get_angle_two_unit_vectors(v1, v2):
    assert np.allclose(np.dot(v1, v1), 1.)
    assert np.allclose(np.dot(v2, v2), 1.)
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))*180./np.pi

def calc_coaxiality_angle(tensor0, tensor1):
    """
        in both tensors the first column is corrosponding to the highest (with sign) eigenvalue
    """
    dot0 = np.dot(tensor0[:,0], tensor1[:,0])
    if dot0 <0: tensor1[:,0] *= -1
    dot1 = np.dot(tensor0[:,1], tensor1[:,1])
    if dot1 <0: tensor1[:,1] *= -1
    
    temp = np.cross(tensor0[:,0], tensor0[:,1])
    if np.dot(temp, tensor0[:,2]) <0: tensor0[:,2] *= -1

    temp = np.cross(tensor1[:,0], tensor1[:,1])
    if np.dot(temp, tensor1[:,2]) <0: tensor1[:,2] *= -1
    
    # res = []
    # for i in range(3):
    #     res.append(np.dot(tensor0[:,i], tensor1[:,i]))
    res = []
    for i in range(3):
        res.append(get_angle_two_unit_vectors(tensor0[:,i], tensor1[:,i]))
    return np.array(res)

def calc_coaxiality_factor(tensor0, tensor1):
    """
        in both tensors the first column is corrosponding to the highest (with sign) eigenvalue
    """
    dot0 = np.dot(tensor0[:,0], tensor1[:,0])
    if dot0 <0: tensor1[:,0] *= -1
    dot1 = np.dot(tensor0[:,1], tensor1[:,1])
    if dot1 <0: tensor1[:,1] *= -1
    
    temp = np.cross(tensor0[:,0], tensor0[:,1])
    if np.dot(temp, tensor0[:,2]) <0: tensor0[:,2] *= -1

    temp = np.cross(tensor1[:,0], tensor1[:,1])
    if np.dot(temp, tensor1[:,2]) <0: tensor1[:,2] *= -1
    
    res = []
    for i in range(3):
        res.append(np.dot(tensor0[:,i], tensor1[:,i]))

    return np.array(res)


root_folder = root_folder_graph_calc + 'data_postproc/'
for data_set_id in datset_ids:
    exp_data = pd.read_csv(root_folder + 'dataset_{}/data_experiment.csv'.format(data_set_id), delimiter=',')
    eps11 = -exp_data['eps11'].values
    A = np.zeros_like(eps11)
    coaxiality_angle = np.zeros((len(eps11), 3))
    coaxiality_factor = np.zeros((len(eps11), 3))
    for i in range(num_load_steps):
        sig = get_stress_tensor(exp_data, i)
        sig_dev = sig - np.trace(sig)/3. * np.eye(3)
        mag_sig_dev = np.sqrt(np.tensordot(sig_dev, sig_dev))
        N_sig_dev = sig_dev / mag_sig_dev


        fab = get_sfabric_tensor(exp_data, i)
        fab_true = 7.5 * (np.copy(fab) - np.eye(3) / 3.)
        mag_fab_true = np.sqrt(np.tensordot(fab_true, fab_true))
        N_fab_true = fab_true / mag_fab_true
        
        sig_eigvals, sig_eigvec = np.linalg.eigh(sig)
        sig_argsort = np.argsort(-sig_eigvals)
        sig_eigvals = sig_eigvals[sig_argsort]
        sig_eigvec = sig_eigvec[:, sig_argsort]

        fab_eigvals, fab_eigvec = np.linalg.eigh(fab)
        fab_argsort = np.argsort(-fab_eigvals)
        fab_eigvals = fab_eigvals[fab_argsort]
        fab_eigvec = fab_eigvec[:, fab_argsort]

        A[i] = abs(np.tensordot(N_fab_true, N_sig_dev))
        coaxiality_angle[i, :] = calc_coaxiality_angle(sig_eigvec, fab_eigvec)
        coaxiality_factor[i, :] = calc_coaxiality_factor(sig_eigvec, fab_eigvec)

    temp = root_folder + 'dataset_{}/figures/'.format(data_set_id)
    plot_x_y([eps11], [A], x_label='Axial Strain', y_label='A',
            add_to_save=temp+'strain_A.png')
    min_val = np.min(coaxiality_factor)
    max_val = np.max(coaxiality_factor)
    rang_val = abs(max_val-min_val)
    plot_x_y([eps11], [coaxiality_factor[:,0]],
              x_label='Axial Strain', y_label='dot product second eigen vector',
              ylim=[min_val-0.1*rang_val, max_val+0.1*rang_val],
              add_to_save=temp+'strain_dotEigVec0.png')
    plot_x_y([eps11], [coaxiality_factor[:,1]],
             x_label='Axial Strain', y_label='dot product second eigen vector',
             ylim=[min_val-0.1*rang_val, max_val+0.1*rang_val],
             add_to_save=temp+'strain_dotEigVec1.png')
    plot_x_y([eps11], [coaxiality_factor[:,2]],
             x_label='Axial Strain', y_label='dot product third eigen vector',
             ylim=[min_val-0.1*rang_val, max_val+0.1*rang_val],
             add_to_save=temp+'strain_dotEigVec2.png')
    min_val = np.min(coaxiality_angle)
    max_val = np.max(coaxiality_angle)
    rang_val = abs(max_val-min_val)
    plot_x_y([eps11, eps11, eps11], [coaxiality_angle[:,0], coaxiality_angle[:,1], coaxiality_angle[:,2]],
              x_label='Axial Strain', y_label='Mismatched Angle [degree]',legend_all=['First Eigenvector', 'Second Eigenvector', 'Third Eigenvector'],
              ylim=[-5, 95],yticks_range=np.arange(0, 91, 30.),
              add_to_save=temp+'strain_MismatchAngle.png')