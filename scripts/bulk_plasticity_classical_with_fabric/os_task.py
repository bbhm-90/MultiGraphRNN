import os
root_to_send = "/Users/bahador/Google\ Drive/Research\ CU/shared_with_Dr_Sun_JHU_work/mismatch_angle/"
datset_ids = range(59)
root_folder_from = '/Users/bahador/projects/paper_drl_short_course_prep_2/sub_graph_training/results/ANN_graphs_bulk_plasticity_classical_with_fabric/data_postproc/'

if not os.path.exists(root_to_send): os.system("mkdir {}".format(root_to_send))


for i in datset_ids:
    temp = root_folder_from + "dataset_{}/figures/strain_MismatchAngle.png".format(i)
    os.system("cp {} {}".format(temp, root_to_send+"strain_MismatchAngle_dataset{}.png".format(i)))

