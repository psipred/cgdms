# Differentiable molecular simulation of proteins with a coarse-grained potential

This repository contains the learned potential, simulation scripts and training code for the paper:

Greener JG and Jones DT, Differentiable molecular simulation can learn all the parameters in a coarse-grained force field for proteins, bioRxiv (2021) - link pending

It provides the `cgdms` Python package which can be used to simulate any protein and reproduce the results in the paper.

## Installation

1. Python 3.6 or later is required.
The software is OS-independent.
2. Install [PyTorch](https://pytorch.org) 1.6 or later as appropriate for your system.
A GPU is not essential but is recommended as simulations are slower on the CPU.
However running on CPU is about 3x slower than GPU depending on hardware, so it is still feasible.
3. Run `pip install cgdms`, which will also install [NumPy](https://numpy.org), [Biopython](https://biopython.org) and [PeptideBuilder](https://github.com/clauswilke/PeptideBuilder) if they are not already present.
The package takes up about 75 MB of disk space.

## Usage

On Unix systems the executable `cgdms` will be added to the path during installation.
On Windows you can call the `bin/cgdms` script with `python` if you can't access the executable.

Run `cgdms -h` to see the help text and `cgdms {mode} -h` to see the help text for each mode.
The modes are described below but there are other options outlined in the help text such as specifying the device to run on, running with a custom parameter set or changing the logging verbosity.

### Generating protein data files

To simulate or calculate the energy of proteins you need to generate files of a certain format.
If you want to use the proteins presented in the paper, the data files are [here](cgdms/protein_data/results).
Otherwise you will need to generate these files:

```bash
cgdms makeinput -i 1CRN.pdb -s 1CRN.ss2 > 1CRN.txt
cat 1CRN.txt
```
```
TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN
CCCCCCCEECCCCCEECCCCCHHHEEEECCCEEEECCCCCCCCCCC
17.047 14.099 3.625 16.967 12.784 4.338 15.685 12.755 5.133 18.551 12.359 5.368
15.115 11.555 5.265 13.856 11.469 6.066 14.164 10.785 7.379 12.841 10.531 4.694
13.488 11.241 8.417 13.66 10.707 9.787 12.269 10.431 10.323 15.126 12.087 10.354
12.019 9.272 10.928 10.646 8.991 11.408 10.654 8.793 12.919 9.947 7.885 9.793
...
```

* `-i` is a well-behaved PDB or mmCIF file.
This means a single protein chain with no missing residues or heavy atoms.
Hetero atoms are ignored and all residues must be standard.
The format is guessed from the file extension, default PDB.
* `-s` is the PSIPRED secondary structure prediction ss2 output file.
If this option is omitted then fully coiled is assumed, which is not recommended, though you could replace that with a secondary structure prediction of your choosing or the known secondary structure depending on your use case.

If you are not interested in the RMSDs logged during the simulation and don't want to start simulation from the native structure, the coordinate lines (which contain coordinates for N/Cα/C/sidechain centroid) are not used.
In this case you can generate your own files with random numbers in place of the coordinates.
This would also apply to sequences where you don't know the native structure.

### Running a simulation

Run a molecular dynamics simulation of a protein in the learned potential:

```bash
cgdms simulate -i 1CRN.txt -o traj.pdb -s predss -n 1.2e7
```
```
    Step        1 / 12000000 - acc  0.005 - vel  0.025 - energy -44.06 ( -21.61 -15.59  -6.86 ) - Cα RMSD  32.59
    Step    10001 / 12000000 - acc  0.005 - vel  0.032 - energy -14.76 ( -11.82   0.46  -3.40 ) - Cα RMSD  32.28
    Step    20001 / 12000000 - acc  0.005 - vel  0.030 - energy  -9.15 (  -8.19   2.15  -3.10 ) - Cα RMSD  31.95
    Step    30001 / 12000000 - acc  0.005 - vel  0.028 - energy  -9.03 ( -10.20   2.22  -1.04 ) - Cα RMSD  31.79
...
```

* `-i` is a protein data file as described above.
* `-o` is the optional output PDB filepath to write the simulation to.
By default snapshots are taken and the energy printed every 10,000 steps but this can be changed with the `-r` flag.
[PULCHRA](https://www.pirx.com/pulchra) can be used to generate all-atom structures from these output files if required.
* `-s` is the starting conformation.
This can be `predss` (extended with predicted secondary structure), `native` (the conformation in the protein data file), `extended` (extended with small random perturbations to the angles), `random` (random in ϕ -180° -> -30°, ψ -180° -> 180°) or `helix` (ϕ -60°, ψ -60°).
* `-n` is the number of simulation steps.
It takes ~36 hours on a GPU to run a simulation of this length, or ~10 ms per time step.
* `-t`, `-c`, `-st`, `-ts` can be used to change the thermostat temperature, thermostat coupling constant, starting temperature and integrator time step respectively.

### Calculating the energy

Calculate the energy of a protein structure in the learned potential:

```bash
cgdms energy -i 1CRN.txt
```
```
-136.122
```

* `-i` is a protein data file as described above.
* `-m` gives an optional number of minimisation steps before returning the energy, default `0`.

Since calculating the energy without minimisation steps is mostly setup, running on the CPU using `-d cpu` is often faster than running on the GPU (~5 s to ~3 s).

### Threading sequences onto a structure

Calculate the energy in the learned potential of a set of sequences threaded onto a structure.

```bash
cgdms thread -i 1CRN.txt -s sequences.txt
```
```
1 -145.448
2 -138.533
3 -142.473
...
```

* `-i` is a protein data file as described above.
* `-s` is a file containing protein sequences, one per line, of the same length as the sequence in the protein data file (that sequence is ignored).
Since lines in the sequence file starting with `>` are ignored, FASTA files can be used provided each sequence is on a single line.
* `-m` gives an optional number of minimisation steps before returning the energy, default `100`.

### Training the system

Train the system.

```bash
cgdms train
```
```
Starting training
Epoch    1 - med train/val RMSD  0.863 /  0.860 over  250 steps
Epoch    2 - med train/val RMSD  0.859 /  0.860 over  250 steps
Epoch    3 - med train/val RMSD  0.856 /  0.854 over  250 steps
...
```

* `-o` is an optional output learned parameter filepath, default `cgdms_params.pt`.

Training takes about 2 months on a decent GPU and is unlikely something you want to do.

### Exploring potentials

The learned potential and information on the interactions can be found in the Python package:

```python
import torch
from cgdms import trained_model_file
params = torch.load(trained_model_file, map_location="cpu")
print(params.keys())
```
```
dict_keys(['distances', 'angles', 'dihedrals', 'optimizer'])
```

* `params["distances"]` has shape `[28961, 140]` corresponding to the 28,960 distance potentials described in the paper and a flat potential used for same atom interactions.
See `cgdms.interactions` for the interaction described by each potential, which has values corresponding to 140 distance bins.
* `params["angles"]` has shape `[5, 20, 140]` corresponding to the 5 bond angles in `cgdms.angles`, the 20 amino acids in `cgdms.aas`, and 140 angle bins.
* `params["dihedrals"]` has shape `[5, 60, 142]` corresponding to the 5 dihedral angles in `cgdms.dihedrals`, the 20 amino acids from `cgdms.aas` in each predicted secondary structure type (ala helix, ala sheet, ala coil, arg helix, etc.), and 140 angle bins with an extra 2 to wrap round and allow periodicity.

## Notes

Running a simulation takes less than 1 GB of GPU memory for any number of steps.
Training a model takes up to 32 GB of GPU memory once the number of steps is fully scaled up to 2,000.

The lists of training and validation PDB chains are available [here](cgdms/datasets) and the protein data files [here](cgdms/protein_data/train_val).

The code in this package is set up to run specific coarse-grained simulations of proteins.
However, the package contains code that could be useful to others wishing to carry out general differentiable simulations with PyTorch.
This includes integrators not used in the paper and not thoroughly tested (velocity-free Verlet, two Langevin implementations), the Andersen thermostat, RMSD with the Kabsch algorithm, and code to apply forces to atoms from bond angle and dihedral angle potentials.

Other software related to differentiable molecular simulation includes [Jax MD](https://github.com/google/jax-md), [TorchMD](https://github.com/torchmd), [DeePMD-kit](https://github.com/deepmodeling/deepmd-kit), [SchNetPack](https://github.com/atomistic-machine-learning/schnetpack), [DiffTaichi](https://github.com/yuanming-hu/difftaichi), [Time Machine](https://github.com/proteneer/timemachine) and [Molly](https://github.com/JuliaMolSim/Molly.jl).
