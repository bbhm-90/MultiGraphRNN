import os
start_id = [5,17,29,41,53]
end_id = [8,21,32,44,57]
assert len(start_id) == len(end_id)
for i,_ in enumerate(start_id):
    os.system("python3 scripts/bulk_plasticity_classical_without_fabric/predict_mean.py {} {}".format(start_id[i], end_id[i]))