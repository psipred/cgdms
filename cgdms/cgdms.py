# Differentiable molecular simulation of proteins with a coarse-grained potential
# Author: Joe G Greener

# biopython and PeptideBuilder are also imported in functions
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.functional import normalize

from itertools import count
from math import pi
import os
from random import gauss, random, shuffle

cgdms_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
dataset_dir = os.path.join(cgdms_dir, "datasets")
train_val_dir = os.path.join(cgdms_dir, "protein_data", "train_val")
trained_model_file = os.path.join(cgdms_dir, "cgdms", "cgdms_params_ep45.pt")

n_bins_pot = 140
n_bins_force = n_bins_pot - 2
n_adjacent = 4

atoms = ["N", "CA", "C", "cent"]

# Last value is the number of atoms in the next residue
angles = [
    ("N", "CA", "C"   , 0), ("CA", "C" , "N"   , 1), ("C", "N", "CA", 2),
    ("N", "CA", "cent", 0), ("C" , "CA", "cent", 0),
]

# Last value is the number of atoms in the next residue
dihedrals = [
    ("C", "N", "CA", "C"   , 3), ("N"   , "CA", "C", "N", 1), ("CA", "C", "N", "CA", 2),
    ("C", "N", "CA", "cent", 3), ("cent", "CA", "C", "N", 1),
]

aas = [
    "A", "R", "N", "D", "C", "E", "Q", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
]
n_aas = len(aas)

one_to_three_aas = {
    "C": "CYS", "D": "ASP", "S": "SER", "Q": "GLN", "K": "LYS",
    "I": "ILE", "P": "PRO", "T": "THR", "F": "PHE", "N": "ASN",
    "G": "GLY", "H": "HIS", "L": "LEU", "R": "ARG", "W": "TRP",
    "A": "ALA", "V": "VAL", "E": "GLU", "Y": "TYR", "M": "MET",
}
three_to_one_aas = {one_to_three_aas[k]: k for k in one_to_three_aas}

aa_masses = {
    "A": 89.09 , "R": 174.2, "N": 132.1, "D": 133.1, "C": 121.2,
    "E": 147.1 , "Q": 146.1, "G": 75.07, "H": 155.2, "I": 131.2,
    "L": 131.2 , "K": 146.2, "M": 149.2, "F": 165.2, "P": 115.1,
    "S": 105.09, "T": 119.1, "W": 204.2, "Y": 181.2, "V": 117.1,
}

ss_types = ["H", "E", "C"]

# Minima in the learned potential after 45 epochs of training
centroid_dists = {
    "A": 1.5575, "R": 4.3575, "N": 2.5025, "D": 2.5025, "C": 2.0825,
    "E": 3.3425, "Q": 3.3775, "G": 1.0325, "H": 3.1675, "I": 2.3975,
    "L": 2.6075, "K": 3.8325, "M": 3.1325, "F": 3.4125, "P": 1.9075,
    "S": 1.9425, "T": 1.9425, "W": 3.9025, "Y": 3.7975, "V": 1.9775,
}

train_proteins = [l.rstrip() for l in open(os.path.join(dataset_dir, "train.txt"))]
val_proteins   = [l.rstrip() for l in open(os.path.join(dataset_dir, "val.txt"  ))]

def get_bin_centres(min_dist, max_dist):
    gap_dist = (max_dist - min_dist) / n_bins_pot
    bcs_pot = [min_dist + i * gap_dist + 0.5 * gap_dist for i in range(n_bins_pot)]
    return bcs_pot[1:-1]

interactions = []
dist_bin_centres = []

# Generate distance interaction list
for i, aa_1 in enumerate(aas):
    for ai, atom_1 in enumerate(atoms):
        for atom_2 in atoms[(ai + 1):]:
            interactions.append(f"{aa_1}_{atom_1}_{aa_1}_{atom_2}_same")
            dist_bin_centres.append(get_bin_centres(0.7, 5.6))
    for aa_2 in aas[i:]:
        for ai, atom_1 in enumerate(atoms):
            atom_iter = atoms if aa_1 != aa_2 else atoms[ai:]
            for atom_2 in atom_iter:
                interactions.append(f"{aa_1}_{atom_1}_{aa_2}_{atom_2}_other")
                dist_bin_centres.append(get_bin_centres(1.0, 15.0))
    for aa_2 in aas:
        for atom_1 in atoms:
            for atom_2 in atoms:
                for ar in range(1, n_adjacent + 1):
                    interactions.append(f"{aa_1}_{atom_1}_{aa_2}_{atom_2}_adj{ar}")
                    dist_bin_centres.append(get_bin_centres(0.7, 14.7))
interactions.append("self_placeholder") # This gets zeroed out during the simulation
dist_bin_centres.append([0.0] * n_bins_force)

gap_ang = (pi - pi / 3) / n_bins_pot
angle_bin_centres = [pi / 3 + i * gap_ang + 0.5 * gap_ang for i in range(n_bins_pot)][1:-1]

gap_dih = (2 * pi) / n_bins_pot
# Two invisible bins on the end imitate periodicity
dih_bin_centres = [-pi + i * gap_dih - 0.5 * gap_dih for i in range(n_bins_pot + 2)][1:-1]

# Report a message if it exceeds the verbosity level
def report(msg, msg_verbosity=0, verbosity=2):
    if msg_verbosity <= verbosity:
        print(msg)

# Read an input data file
# The protein sequence is read from the file but will overrule the file if provided
def read_input_file(fp, seq="", device="cpu"):
    with open(fp) as f:
        lines = f.readlines()
        if seq == "":
            seq = lines[0].rstrip()
        ss_pred = lines[1].rstrip()
        assert len(seq) == len(ss_pred), f"Sequence length is {len(seq)} but SS prediction length is {len(ss_pred)}"
    seq_info = []
    for i in range(len(seq)):
        for atom in atoms:
            seq_info.append((i, atom))
    n_atoms = len(seq_info)
    native_coords = torch.tensor(np.loadtxt(fp, skiprows=2), dtype=torch.float,
                                    device=device).view(n_atoms, 3)

    inters = torch.ones(n_atoms, n_atoms, dtype=torch.long, device=device) * -1
    for i in range(n_atoms):
        inters[i, i] = len(interactions) - 1 # Placeholder for same atom
        for j in range(i):
            res_sep = abs(seq_info[i][0] - seq_info[j][0])
            if 1 <= res_sep <= n_adjacent:
                # Due to known ordering we know that the order of residues is j->i
                info_1, info_2 = seq_info[j], seq_info[i]
            else:
                # Sort by amino acid index then by atom
                info_1, info_2 = sorted([seq_info[i], seq_info[j]],
                                        key=lambda x : (aas.index(seq[x[0]]), atoms.index(x[1])))
            inter = f"{seq[info_1[0]]}_{info_1[1]}_{seq[info_2[0]]}_{info_2[1]}"
            if res_sep == 0:
                inter += "_same"
            elif res_sep <= n_adjacent:
                inter += f"_adj{res_sep}"
            else:
                inter += "_other"
            inter_i = interactions.index(inter)
            inters[i, j] = inter_i
            inters[j, i] = inter_i
    inters_flat = inters.view(n_atoms * n_atoms)

    masses = []
    for i, r in enumerate(seq):
        mass_CA = 13.0 # Includes H
        mass_N = 15.0 # Includes amide H
        if i == 0:
            mass_N += 2.0 # Add charged N-terminus
        mass_C = 28.0 # Includes carbonyl O
        if i == len(seq) - 1:
            mass_C += 16.0 # Add charged C-terminus
        mass_cent = aa_masses[r] - 74.0 # Subtract non-centroid section
        if r == "G":
            mass_cent += 10.0 # Make glycine artificially heavier
        masses.append(mass_N)
        masses.append(mass_CA)
        masses.append(mass_C)
        masses.append(mass_cent)
    masses = torch.tensor(masses, device=device)

    # Different angle potentials for each residue
    inters_ang = torch.tensor([aas.index(r) for r in seq], dtype=torch.long, device=device)

    # Different dihedral potentials for each residue and predicted secondary structure type
    inters_dih = torch.tensor([aas.index(r) * len(ss_types) + ss_types.index(s) for r, s in zip(seq, ss_pred)],
                                dtype=torch.long, device=device)

    return native_coords, inters_flat, inters_ang, inters_dih, masses, seq

# Read an input data file and thread a new sequence onto it
def read_input_file_threaded(fp, seq, device="cpu"):
    coords, inters_flat, inters_ang, inters_dih, masses, seq = read_input_file(fp, seq, device=device)

    # Move centroids out to minimum distances for that sequence
    ind_ca, ind_cent = atoms.index("CA"), atoms.index("cent")
    for i, r in enumerate(seq):
        ca_cent_diff = coords[i * len(atoms) + ind_cent] - coords[i * len(atoms) + ind_ca]
        ca_cent_unitvec = ca_cent_diff / ca_cent_diff.norm()
        coords[i * len(atoms) + ind_cent] = coords[i * len(atoms) + ind_ca] + centroid_dists[r] * ca_cent_unitvec

    return coords, inters_flat, inters_ang, inters_dih, masses, seq

# Read a dataset of input files
class ProteinDataset(Dataset):
    def __init__(self, pdbids, coord_dir, device="cpu"):
        self.pdbids = pdbids
        self.coord_dir = coord_dir
        self.set_size = len(pdbids)
        self.device = device

    def __len__(self):
        return self.set_size

    def __getitem__(self, index):
        fp = os.path.join(self.coord_dir, self.pdbids[index] + ".txt")
        return read_input_file(fp, device=self.device)

# Differentiable molecular simulation of proteins with a coarse-grained potential
class Simulator(torch.nn.Module):
    def __init__(self, ff_distances, ff_angles, ff_dihedrals):
        super(Simulator, self).__init__()
        self.ff_distances = torch.nn.Parameter(ff_distances)
        self.ff_angles    = torch.nn.Parameter(ff_angles)
        self.ff_dihedrals = torch.nn.Parameter(ff_dihedrals)

    def forward(self,
                coords,
                inters_flat,
                inters_ang,
                inters_dih,
                masses,
                seq,
                native_coords,
                n_steps,
                integrator="vel", # vel/no_vel/min/langevin/langevin_simple
                timestep=0.02,
                start_temperature=0.1,
                thermostat_const=0.0, # Set to 0.0 to run without a thermostat (NVE ensemble)
                temperature=0.0, # The effective temperature of the thermostat
                sim_filepath=None, # Output PDB file to write to or None to not write out
                energy=False, # Return the energy at the end of the simulation
                report_n=10_000, # Print and write PDB every report_n steps
                verbosity=2, # 0 for epoch info, 1 for protein info, 2 for simulation step info
        ):

        assert integrator in ("vel", "no_vel", "min", "langevin", "langevin_simple"), f"Invalid integrator {integrator}"
        device = coords.device
        batch_size, n_atoms = masses.size(0), masses.size(1)
        n_res = n_atoms // len(atoms)
        dist_bin_centres_tensor = torch.tensor(dist_bin_centres, device=device)
        pair_centres_flat = dist_bin_centres_tensor.index_select(0, inters_flat[0]).unsqueeze(0).expand(batch_size, -1, -1)
        pair_pots_flat = self.ff_distances.index_select(0, inters_flat[0]).unsqueeze(0).expand(batch_size, -1, -1)
        angle_bin_centres_tensor = torch.tensor(angle_bin_centres, device=device)
        angle_centres_flat = angle_bin_centres_tensor.unsqueeze(0).unsqueeze(0).expand(batch_size, n_res, -1)
        angle_pots_flat = self.ff_angles.index_select(1, inters_ang[0]).unsqueeze(0).expand(batch_size, -1, -1, -1)
        dih_bin_centres_tensor = torch.tensor(dih_bin_centres, device=device)
        dih_centres_flat = dih_bin_centres_tensor.unsqueeze(0).unsqueeze(0).expand(batch_size, n_res - 1, -1)
        dih_pots_flat = self.ff_dihedrals.index_select(1, inters_dih[0]).unsqueeze(0).expand(batch_size, -1, -1, -1)
        native_coords_ca = native_coords.view(batch_size, n_res, 3 * len(atoms))[0, :, 3:6]
        model_n = 0

        if integrator == "vel" or integrator == "langevin" or integrator == "langevin_simple":
            vels = torch.randn(coords.shape, device=device) * start_temperature
            accs_last = torch.zeros(coords.shape, device=device)
        elif integrator == "no_vel":
            coords_last = coords.clone() + torch.randn(coords.shape, device=device) * start_temperature * timestep

        # The step the energy is return on is not used for simulation so we add an extra step
        if energy:
            n_steps += 1

        for i in range(n_steps):
            if integrator == "vel":
                coords = coords + vels * timestep + 0.5 * accs_last * timestep * timestep
            elif integrator == "langevin":
                # From Gronbech-Jensen 2013
                alpha, twokbT = thermostat_const, temperature
                beta = np.sqrt(twokbT * alpha * timestep) * torch.randn(vels.shape, device=device)
                b = 1.0 / (1.0 + (alpha * timestep) / (2 * masses.unsqueeze(2)))
                coords_last = coords
                coords = coords + b * timestep * vels + 0.5 * b * (timestep ** 2) * accs_last + 0.5 * b * timestep * beta / masses.unsqueeze(2)
            elif integrator == "langevin_simple":
                coords = coords + vels * timestep + 0.5 * accs_last * timestep * timestep

            # See https://arxiv.org/pdf/1401.1181.pdf for derivation of forces
            printing = verbosity >= 2 and i % report_n == 0
            returning_energy = energy and i == n_steps - 1
            if printing or returning_energy:
                dist_energy = torch.zeros(1, device=device)
                angle_energy = torch.zeros(1, device=device)
                dih_energy = torch.zeros(1, device=device)

            # Add pairwise distance forces
            crep = coords.unsqueeze(1).expand(-1, n_atoms, -1, -1)
            diffs = crep - crep.transpose(1, 2)
            dists = diffs.norm(dim=3)
            dists_flat = dists.view(batch_size, n_atoms * n_atoms)
            dists_from_centres = pair_centres_flat - dists_flat.unsqueeze(2).expand(-1, -1, n_bins_force)
            dist_bin_inds = dists_from_centres.abs().argmin(dim=2).unsqueeze(2)
            # Force is gradient of potential
            # So it is proportional to difference of previous and next value of potential
            pair_forces_flat = 0.5 * (pair_pots_flat.gather(2, dist_bin_inds) - pair_pots_flat.gather(2, dist_bin_inds + 2))
            # Specify minimum to prevent division by zero errors
            norm_diffs = diffs / dists.clamp(min=0.01).unsqueeze(3)
            pair_accs = (pair_forces_flat.view(batch_size, n_atoms, n_atoms)).unsqueeze(3) * norm_diffs
            accs = pair_accs.sum(dim=1) / masses.unsqueeze(2)
            if printing or returning_energy:
                dist_energy += 0.5 * pair_pots_flat.gather(2, dist_bin_inds + 1).sum()

            atom_coords = coords.view(batch_size, n_res, 3 * len(atoms))
            atom_accs = torch.zeros(batch_size, n_res, 3 * len(atoms), device=device)
            # Angle forces
            # across_res is the number of atoms in the next residue, starting from atom_3
            for ai, (atom_1, atom_2, atom_3, across_res) in enumerate(angles):
                ai_1, ai_2, ai_3 = atoms.index(atom_1), atoms.index(atom_2), atoms.index(atom_3)
                if across_res == 0:
                    ba = atom_coords[:, :  , (ai_1 * 3):(ai_1 * 3 + 3)] - atom_coords[:, :  , (ai_2 * 3):(ai_2 * 3 + 3)]
                    bc = atom_coords[:, :  , (ai_3 * 3):(ai_3 * 3 + 3)] - atom_coords[:, :  , (ai_2 * 3):(ai_2 * 3 + 3)]
                    # Use residue potential according to central atom
                    angle_pots_to_use = angle_pots_flat[:, ai, :]
                elif across_res == 1:
                    ba = atom_coords[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)] - atom_coords[:, :-1, (ai_2 * 3):(ai_2 * 3 + 3)]
                    bc = atom_coords[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)] - atom_coords[:, :-1, (ai_2 * 3):(ai_2 * 3 + 3)]
                    angle_pots_to_use = angle_pots_flat[:, ai, :-1]
                elif across_res == 2:
                    ba = atom_coords[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)] - atom_coords[:, 1: , (ai_2 * 3):(ai_2 * 3 + 3)]
                    bc = atom_coords[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)] - atom_coords[:, 1: , (ai_2 * 3):(ai_2 * 3 + 3)]
                    angle_pots_to_use = angle_pots_flat[:, ai, 1:]
                ba_norms = ba.norm(dim=2)
                bc_norms = bc.norm(dim=2)
                angs = torch.acos((ba * bc).sum(dim=2) / (ba_norms * bc_norms))
                n_angles = n_res if across_res == 0 else n_res - 1
                angles_from_centres = angle_centres_flat[:, :n_angles] - angs.unsqueeze(2)
                angle_bin_inds = angles_from_centres.abs().argmin(dim=2).unsqueeze(2)
                angle_forces = 0.5 * (angle_pots_to_use.gather(2, angle_bin_inds) - angle_pots_to_use.gather(2, angle_bin_inds + 2))
                cross_ba_bc = torch.cross(ba, bc, dim=2)
                fa = angle_forces * normalize(torch.cross( ba, cross_ba_bc, dim=2), dim=2) / ba_norms.unsqueeze(2)
                fc = angle_forces * normalize(torch.cross(-bc, cross_ba_bc, dim=2), dim=2) / bc_norms.unsqueeze(2)
                fb = -fa -fc
                if across_res == 0:
                    atom_accs[:, :  , (ai_1 * 3):(ai_1 * 3 + 3)] += fa
                    atom_accs[:, :  , (ai_2 * 3):(ai_2 * 3 + 3)] += fb
                    atom_accs[:, :  , (ai_3 * 3):(ai_3 * 3 + 3)] += fc
                elif across_res == 1:
                    atom_accs[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)] += fa
                    atom_accs[:, :-1, (ai_2 * 3):(ai_2 * 3 + 3)] += fb
                    atom_accs[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)] += fc
                elif across_res == 2:
                    atom_accs[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)] += fa
                    atom_accs[:, 1: , (ai_2 * 3):(ai_2 * 3 + 3)] += fb
                    atom_accs[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)] += fc
                if printing or returning_energy:
                    angle_energy += angle_pots_to_use.gather(2, angle_bin_inds + 1).sum()

            # Dihedral forces
            # across_res is the number of atoms in the next residue, starting from atom_4
            for di, (atom_1, atom_2, atom_3, atom_4, across_res) in enumerate(dihedrals):
                ai_1, ai_2, ai_3, ai_4 = atoms.index(atom_1), atoms.index(atom_2), atoms.index(atom_3), atoms.index(atom_4)
                if across_res == 1:
                    ab = atom_coords[:, :-1, (ai_2 * 3):(ai_2 * 3 + 3)] - atom_coords[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)]
                    bc = atom_coords[:, :-1, (ai_3 * 3):(ai_3 * 3 + 3)] - atom_coords[:, :-1, (ai_2 * 3):(ai_2 * 3 + 3)]
                    cd = atom_coords[:, 1: , (ai_4 * 3):(ai_4 * 3 + 3)] - atom_coords[:, :-1, (ai_3 * 3):(ai_3 * 3 + 3)]
                    # Use residue potential according to central atom
                    dih_pots_to_use = dih_pots_flat[:, di, :-1]
                elif across_res == 2:
                    ab = atom_coords[:, :-1, (ai_2 * 3):(ai_2 * 3 + 3)] - atom_coords[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)]
                    bc = atom_coords[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)] - atom_coords[:, :-1, (ai_2 * 3):(ai_2 * 3 + 3)]
                    cd = atom_coords[:, 1: , (ai_4 * 3):(ai_4 * 3 + 3)] - atom_coords[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)]
                    dih_pots_to_use = dih_pots_flat[:, di, 1:]
                elif across_res == 3:
                    ab = atom_coords[:, 1: , (ai_2 * 3):(ai_2 * 3 + 3)] - atom_coords[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)]
                    bc = atom_coords[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)] - atom_coords[:, 1: , (ai_2 * 3):(ai_2 * 3 + 3)]
                    cd = atom_coords[:, 1: , (ai_4 * 3):(ai_4 * 3 + 3)] - atom_coords[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)]
                    dih_pots_to_use = dih_pots_flat[:, di, 1:]
                cross_ab_bc = torch.cross(ab, bc, dim=2)
                cross_bc_cd = torch.cross(bc, cd, dim=2)
                bc_norms = bc.norm(dim=2).unsqueeze(2)
                dihs = torch.atan2(
                    torch.sum(torch.cross(cross_ab_bc, cross_bc_cd, dim=2) * bc / bc_norms, dim=2),
                    torch.sum(cross_ab_bc * cross_bc_cd, dim=2)
                )
                dihs_from_centres = dih_centres_flat - dihs.unsqueeze(2)
                dih_bin_inds = dihs_from_centres.abs().argmin(dim=2).unsqueeze(2)
                dih_forces = 0.5 * (dih_pots_to_use.gather(2, dih_bin_inds) - dih_pots_to_use.gather(2, dih_bin_inds + 2))
                fa = dih_forces * normalize(-cross_ab_bc, dim=2) / ab.norm(dim=2).unsqueeze(2)
                fd = dih_forces * normalize( cross_bc_cd, dim=2) / cd.norm(dim=2).unsqueeze(2)
                # Forces on the middle atoms have to keep the sum of torques null
                # Forces taken from http://www.softberry.com/freedownloadhelp/moldyn/description.html
                fb = ((ab * -bc) / (bc_norms ** 2) - 1) * fa - ((cd * -bc) / (bc_norms ** 2)) * fd
                fc = -fa - fb - fd
                if across_res == 1:
                    atom_accs[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)] += fa
                    atom_accs[:, :-1, (ai_2 * 3):(ai_2 * 3 + 3)] += fb
                    atom_accs[:, :-1, (ai_3 * 3):(ai_3 * 3 + 3)] += fc
                    atom_accs[:, 1: , (ai_4 * 3):(ai_4 * 3 + 3)] += fd
                elif across_res == 2:
                    atom_accs[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)] += fa
                    atom_accs[:, :-1, (ai_2 * 3):(ai_2 * 3 + 3)] += fb
                    atom_accs[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)] += fc
                    atom_accs[:, 1: , (ai_4 * 3):(ai_4 * 3 + 3)] += fd
                elif across_res == 3:
                    atom_accs[:, :-1, (ai_1 * 3):(ai_1 * 3 + 3)] += fa
                    atom_accs[:, 1: , (ai_2 * 3):(ai_2 * 3 + 3)] += fb
                    atom_accs[:, 1: , (ai_3 * 3):(ai_3 * 3 + 3)] += fc
                    atom_accs[:, 1: , (ai_4 * 3):(ai_4 * 3 + 3)] += fd
                if printing or returning_energy:
                    dih_energy += dih_pots_to_use.gather(2, dih_bin_inds + 1).sum()

            accs += atom_accs.view(batch_size, n_atoms, 3) / masses.unsqueeze(2)

            # Shortcut to return energy at a given step
            if returning_energy:
                return dist_energy + angle_energy + dih_energy

            if integrator == "vel":
                vels = vels + 0.5 * (accs_last + accs) * timestep
                accs_last = accs
            elif integrator == "no_vel":
                coords_next = 2 * coords - coords_last + accs * timestep * timestep
                coords_last = coords
                coords = coords_next
            elif integrator == "langevin":
                # From Gronbech-Jensen 2013
                vels = vels + 0.5 * timestep * (accs_last + accs) - alpha * (coords - coords_last) / masses.unsqueeze(2) + beta / masses.unsqueeze(2)
                accs_last = accs
            elif integrator == "langevin_simple":
                gamma, twokbT = thermostat_const, temperature
                accs = accs + (-gamma * vels + np.sqrt(gamma * twokbT) * torch.randn(vels.shape, device=device)) / masses.unsqueeze(2)
                vels = vels + 0.5 * (accs_last + accs) * timestep
                accs_last = accs
            elif integrator == "min":
                coords = coords + accs * 0.1

            # Apply thermostat
            if integrator in ("vel", "no_vel") and thermostat_const > 0.0:
                thermostat_prob = timestep / thermostat_const
                for ai in range(n_atoms):
                    if random() < thermostat_prob:
                        if integrator == "vel":
                            # Actually this should be divided by the mass
                            new_vel = torch.randn(3, device=device) * temperature
                            vels[0, ai] = new_vel
                        elif integrator == "no_vel":
                            new_diff = torch.randn(3, device=device) * temperature * timestep
                            coords_last[0, ai] = coords[0, ai] - new_diff

            if printing:
                total_energy = dist_energy + angle_energy + dih_energy
                out_line = "    Step {:8} / {} - acc {:6.3f} {}- energy {:6.2f} ( {:6.2f} {:6.2f} {:6.2f} ) - Cα RMSD {:6.2f}".format(
                    i + 1, n_steps, torch.mean(accs.norm(dim=2)).item(),
                    "- vel {:6.3f} ".format(torch.mean(vels.norm(dim=2)).item()) if integrator in ("vel", "langevin", "langevin_simple") else "",
                    total_energy.item(), dist_energy.item(), angle_energy.item(), dih_energy.item(),
                    rmsd(coords.view(batch_size, n_res, 3 * len(atoms))[0, :, 3:6], native_coords_ca)[0].item())
                report(out_line, 2, verbosity)

            if sim_filepath and i % report_n == 0:
                model_n += 1
                with open(sim_filepath, "a") as of:
                    of.write("MODEL {:>8}\n".format(model_n))
                    for ri, r in enumerate(seq):
                        for ai, atom in enumerate(atoms):
                            of.write("ATOM   {:>4}  {:<2}  {:3} A{:>4}    {:>8.3f}{:>8.3f}{:>8.3f}  1.00  0.00          {:>2}  \n".format(
                                len(atoms) * ri + ai + 1, atom[:2].upper(),
                                one_to_three_aas[r], ri + 1,
                                coords[0, len(atoms) * ri + ai, 0].item(),
                                coords[0, len(atoms) * ri + ai, 1].item(),
                                coords[0, len(atoms) * ri + ai, 2].item(),
                                atom[0].upper()))
                    of.write("ENDMDL\n")

        return coords

# RMSD between two sets of coordinates with shape (n_atoms, 3) using the Kabsch algorithm
# Returns the RMSD and whether convergence was reached
def rmsd(c1, c2):
    device = c1.device
    r1 = c1.transpose(0, 1)
    r2 = c2.transpose(0, 1)
    P = r1 - r1.mean(1).view(3, 1)
    Q = r2 - r2.mean(1).view(3, 1)
    cov = torch.matmul(P, Q.transpose(0, 1))
    try:
        U, S, V = torch.svd(cov)
    except RuntimeError:
        report("  SVD failed to converge", 0)
        return torch.tensor([20.0], device=device), False
    d = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, torch.det(torch.matmul(V, U.transpose(0, 1)))]
    ], device=device)
    rot = torch.matmul(torch.matmul(V, d), U.transpose(0, 1))
    rot_P = torch.matmul(rot, P)
    diffs = rot_P - Q
    msd = (diffs ** 2).sum() / diffs.size(1)
    return msd.sqrt(), True

# Generate starting coordinates
# conformation is extended/predss/random/helix
def starting_coords(seq, conformation="extended", input_file="", device="cpu"):
    import PeptideBuilder

    coords = torch.zeros(len(seq) * len(atoms), 3, device=device)
    backbone_atoms = ("N", "CA", "C", "O")
    ss_phis = {"C": -120.0, "H": -60.0, "E": -120.0}
    ss_psis = {"C":  140.0, "H": -60.0, "E":  140.0}

    if conformation == "predss":
        with open(input_file) as f:
            ss_pred = f.readlines()[1].rstrip()
    for i, r in enumerate(seq):
        r_to_use = "A" if r == "G" else r
        if i == 0:
            structure = PeptideBuilder.initialize_res(r_to_use)
        elif conformation == "predss":
            structure = PeptideBuilder.add_residue(structure, r_to_use, ss_phis[ss_pred[i]], ss_psis[ss_pred[i]])
        elif conformation == "random":
            # ϕ can be -180° -> -30°, ψ can be anything
            phi = -180 + random() * 150
            psi = -180 + random() * 360
            structure = PeptideBuilder.add_residue(structure, r_to_use, phi, psi)
        elif conformation == "helix":
            structure = PeptideBuilder.add_residue(structure, r_to_use, ss_phis["H"], ss_psis["H"])
        elif conformation == "extended":
            coil_level = 30.0
            phi = -120.0 + gauss(0.0, coil_level)
            psi =  140.0 + gauss(0.0, coil_level)
            structure = PeptideBuilder.add_residue(structure, r_to_use, phi, psi)
        else:
            raise(AssertionError(f"Invalid conformation {conformation}"))
        for ai, atom in enumerate(atoms):
            if atom == "cent":
                coords[len(atoms) * i + ai] = torch.tensor(
                    [at.coord for at in structure[0]["A"][i + 1] if at.name not in backbone_atoms],
                    dtype=torch.float, device=device).mean(dim=0)
            else:
                coords[len(atoms) * i + ai] = torch.tensor(structure[0]["A"][i + 1][atom].coord,
                                                            dtype=torch.float, device=device)
    return coords

# Print a protein data file from a PDB/mmCIF file and an optional PSIPRED ss2 file
def print_input_file(structure_file, ss2_file=None):
    extension = os.path.basename(structure_file).rsplit(".", 1)[-1].lower()
    if extension in ("cif", "mmcif"):
        from Bio.PDB import MMCIFParser
        parser = MMCIFParser()
    else:
        from Bio.PDB import PDBParser
        parser = PDBParser()
    struc = parser.get_structure("", structure_file)

    seq = ""
    coords = []
    for chain in struc[0]:
        for res in chain:
            # Skip hetero and water residues
            if res.id[0] != " ":
                continue
            seq += three_to_one_aas[res.get_resname()]
            if res.get_resname() == "GLY":
                # Extend vector of length 1 Å from Cα to act as fake centroid
                d = res["CA"].get_coord() - res["C"].get_coord() + res["CA"].get_coord() - res["N"].get_coord()
                coord_cent = res["CA"].get_coord() + d / np.linalg.norm(d)
            else:
                # Centroid coordinates of sidechain heavy atoms
                atom_coords = []
                for atom in res:
                    if atom.get_name() not in ("N", "CA", "C", "O") and atom.element != "H":
                        atom_coords.append(atom.get_coord())
                coord_cent = np.array(atom_coords).mean(0)
            coords.append([res["N"].get_coord(), res["CA"].get_coord(), res["C"].get_coord(), coord_cent])

    print(seq)
    if ss2_file:
        # Extract 3-state secondary structure prediction from PSIPRED ss2 output file
        ss_pred = ""
        with open(ss2_file) as f:
            for line in f:
                if len(line.rstrip()) > 0 and not line.startswith("#"):
                    ss_pred += line.split()[2]
        assert len(seq) == len(ss_pred), f"Sequence length is {len(seq)} but SS prediction length is {len(ss_pred)}"
        print(ss_pred)
    else:
        print("C" * len(seq))

    def coord_str(coord):
        return " ".join([str(round(c, 3)) for c in coord])

    for coord_n, coord_ca, coord_c, coord_cent in coords:
        print(f"{coord_str(coord_n)} {coord_str(coord_ca)} {coord_str(coord_c)} {coord_str(coord_cent)}")

def train(model_filepath, device="cpu", verbosity=0):
    max_n_steps = 2_000
    learning_rate = 1e-4
    n_accumulate = 100

    simulator = Simulator(
        torch.zeros(len(interactions), n_bins_pot, device=device),
        torch.zeros(len(angles), n_aas, n_bins_pot, device=device),
        torch.zeros(len(dihedrals), n_aas * len(ss_types), n_bins_pot + 2, device=device)
    )

    train_set = ProteinDataset(train_proteins, train_val_dir, device=device)
    val_set   = ProteinDataset(val_proteins  , train_val_dir, device=device)

    optimizer = torch.optim.Adam(simulator.parameters(), lr=learning_rate)

    report("Starting training", 0, verbosity)
    for ei in count(start=0, step=1):
        # After 37 epochs reset the optimiser with a lower learning rate
        if ei == 37:
            optimizer = torch.optim.Adam(simulator.parameters(), lr=learning_rate / 2)

        train_rmsds, val_rmsds = [], []
        n_steps = min(250 * ((ei // 5) + 1), max_n_steps) # Scale up n_steps over epochs
        train_inds = list(range(len(train_set)))
        val_inds   = list(range(len(val_set)))
        shuffle(train_inds)
        shuffle(val_inds)
        simulator.train()
        optimizer.zero_grad()
        for i, ni in enumerate(train_inds):
            native_coords, inters_flat, inters_ang, inters_dih, masses, seq = train_set[ni]
            coords = simulator(native_coords.unsqueeze(0), inters_flat.unsqueeze(0),
                                inters_ang.unsqueeze(0), inters_dih.unsqueeze(0), masses.unsqueeze(0),
                                seq, native_coords.unsqueeze(0), n_steps, verbosity=verbosity)
            loss, passed = rmsd(coords[0], native_coords)
            train_rmsds.append(loss.item())
            if passed:
                loss_log = torch.log(1.0 + loss)
                loss_log.backward()
            report("  Training   {:4} / {:4} - RMSD {:6.2f} over {:4} steps and {:3} residues".format(
                    i + 1, len(train_set), loss.item(), n_steps, len(seq)), 1, verbosity)
            if (i + 1) % n_accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
        simulator.eval()
        with torch.no_grad():
            for i, ni in enumerate(val_inds):
                native_coords, inters_flat, inters_ang, inters_dih, masses, seq = val_set[ni]
                coords = simulator(native_coords.unsqueeze(0), inters_flat.unsqueeze(0),
                                    inters_ang.unsqueeze(0), inters_dih.unsqueeze(0), masses.unsqueeze(0),
                                    seq, native_coords.unsqueeze(0), n_steps, verbosity=verbosity)
                loss, passed = rmsd(coords[0], native_coords)
                val_rmsds.append(loss.item())
                report("  Validation {:4} / {:4} - RMSD {:6.2f} over {:4} steps and {:3} residues".format(
                        i + 1, len(val_set), loss.item(), n_steps, len(seq)), 1, verbosity)
        torch.save({"distances": simulator.ff_distances.data,
                    "angles"   : simulator.ff_angles.data,
                    "dihedrals": simulator.ff_dihedrals.data,
                    "optimizer": optimizer.state_dict()},
                    model_filepath)
        report("Epoch {:4} - med train/val RMSD {:6.3f} / {:6.3f} over {:4} steps".format(
                ei + 1, np.median(train_rmsds), np.median(val_rmsds), n_steps), 0, verbosity)
