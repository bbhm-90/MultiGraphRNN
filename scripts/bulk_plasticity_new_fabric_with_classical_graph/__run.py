import os
start_id = [5,17,29,41,53]
end_id = [8,21,32,44,57]
assert len(start_id) == len(end_id)
for i,_ in enumerate(start_id):
    os.system("python3 scripts/ANN_graphs_bulk_plasticity_new_fabric_with_classical_graph/predict_mean.py {} {}".format(start_id[i], end_id[i]))