import torch


def steric_repulsion_energy(
    protein_pos,
    ligand_pos,
    batch_protein,
    batch_ligand,
    cutoff=1.5,
    reduction='mean',
):
    """
    protein_pos: [Np, 3]
    ligand_pos: [Nl, 3]
    batch_protein: [Np]
    batch_ligand: [Nl]

    回傳每個 complex 的 steric energy，最後再做 reduction
    """

    device = protein_pos.device
    num_graphs = int(batch_ligand.max().item()) + 1
    energies = []

    for g in range(num_graphs):
        p = protein_pos[batch_protein == g]   # [Np_g, 3]
        l = ligand_pos[batch_ligand == g]     # [Nl_g, 3]

        if p.numel() == 0 or l.numel() == 0:
            energies.append(torch.zeros((), device=device))
            continue

        # pairwise distance: [Np_g, Nl_g]
        dists = torch.cdist(p, l, p=2)

        # 若距離小於 cutoff，產生懲罰
        penalty = torch.relu(cutoff - dists) ** 2
        energy_g = penalty.sum()
        energies.append(energy_g)

    energies = torch.stack(energies, dim=0)  # [B]

    if reduction == 'mean':
        return energies.mean()
    elif reduction == 'sum':
        return energies.sum()
    else:
        return energies