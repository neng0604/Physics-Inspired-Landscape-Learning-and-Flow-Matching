from .steric import steric_repulsion_energy


def compute_binding_energy(
    protein_pos,
    ligand_pos,
    batch_protein,
    batch_ligand,
    cutoff=1.5,
):
    e_steric = steric_repulsion_energy(
        protein_pos=protein_pos,
        ligand_pos=ligand_pos,
        batch_protein=batch_protein,
        batch_ligand=batch_ligand,
        cutoff=cutoff,
        reduction='mean',
    )

    return {
        'E_total': e_steric,
        'E_steric': e_steric,
    }