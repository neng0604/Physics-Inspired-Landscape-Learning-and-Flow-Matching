import torch
import numpy as np

from models.molopt_score_model_guide import AtomCountPredictor

def denormalize_label(atom_num_normalized, min_val=3, max_val=106):
    return atom_num_normalized * (max_val - min_val) + min_val

def predict_atom_num(data, device, ckpt_path, n_data, atom_num_std):

    model = AtomCountPredictor().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    atom_num_pred_normalized = model(data).item()
    atom_num_pred_normalized = atom_num_pred_normalized + np.random.normal(
        loc=0, scale=atom_num_std, size=n_data
    )
    atom_num_pred_normalized[atom_num_pred_normalized < 0] = 0
    atom_num_pred = denormalize_label(atom_num_pred_normalized)
    return np.round(atom_num_pred).astype(int).tolist()
