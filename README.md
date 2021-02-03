# Differentiable molecular simulation of proteins with a coarse-grained potential

This repository contains the learned potential, simulation scripts and training code for the paper:

...

## Installation

1. Python 3.6 or later is required.
The software is OS-independent.
2. Install [PyTorch](https://pytorch.org) 1.6 or later as appropriate for your system.
A GPU is not essential but is highly recommended as simulations are much slower on the CPU.
3. Run `pip install cgdms`, which will also install [NumPy](https://numpy.org), [Biopython](https://biopython.org) and [PeptideBuilder](https://github.com/clauswilke/PeptideBuilder) if they are not already present.
The package takes up about 75 MB of disk space.

## Usage

On Unix systems the executable `cgdms` will be added to the path during installation.
On Windows you can call the `bin/cgdms` script with `python` if you can't access the executable.

Run `cgdms -h` to see the help text and `cgdms {mode} -h` to see the help text for each mode.
The modes are described below but there are other options outlined in the help text such as specifying the device to run on, running with a custom parameter set or changing the logging verbosity.

### Generating protein data files

To simulate or score proteins you need to generate files of a certain format.
If you want to use the proteins used in the paper, the data files are at...
Otherwise you will need to generate these files:

```bash
cgdms makeinput -i protein.pdb -s protein.ss2
```
```
...
```

* `-i` is a well-behaved PDB or mmCIF file.
This means a single protein chain with no missing residues or heavy atoms.
Hetero atoms are ignored and all residues must be standard.
The format is guessed from the file extension, default PDB.
* `-s` is the PSIPRED secondary structure prediction ss2 output file.
If this option is omitted then fully coiled is assumed, which is not recommended, though you could replace that with a secondary structure prediction of your choosing or the known secondary structure.

If you are not interested in the RMSDs logged during the simulation and don't want to start simulation from the native structure, the coordinate lines are not used.
In this case you can generate your own files with random numbers in place of the coordinates.
This would also apply to sequences where you don't know the native structure.

### Running a simulation

Run a molecular dynamics simulation of a protein in the learned potential:

```bash
cgdms simulate -i protein.txt -o traj.pdb -s predss -n 1.2e7
```
```
...
```

* `-i` is a protein data file as described above.
* `-o` is the optional output PDB filepath to write the simulation to.
By default snapshots are taken and the energy printed every 10,000 steps but this can be changed with the `-r` flag.
PULCHRA...
* `-s` is the starting conformation.
This can be `predss` (extended with predicted secondary structure), `native` (the conformation in the protein data file), `extended` (extended with small random perturbations to the angles), `random` (random in ϕ -180° -> -30°, ψ -180° -> 180°) or `helix` (ϕ -60°, ψ -60°).
* `-n` is the number of simulation steps.
It takes ~36 hours on a GPU to run a simulation of this length.
* `-t`, `-c`, `-st`, `-ts` can be used to change the thermostat temperature, thermostat coupling constant, starting temperature and integrator time step respectively.

### Scoring a structure

Score a protein structure in the learned potential:

```bash
cgdms -i protein.txt
```
```
...
```

* `-i` is a protein data file as described above.
* `-m` gives an optional number of minimisation steps before returning the score, default `0`.

### Threading sequences onto a structure

Thread a set of sequences onto a structure and score them.

```bash
cgdms thread -i protein.txt -s sequences.txt
```
```
...
```

* `-i` is a protein data file as described above.
* `-s` is a file containing protein sequences, one per line, of the same length as the sequence in the protein data file (that sequence is ignored).
Since lines starting with `>` are ignored, FASTA files can be used provided each sequence is on a single line.
* `-m` gives an optional number of minimisation steps before returning the score, default `100`.

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
See `cgdms.interactions` for the interactions described by each potential, which has values corresponding to 140 distance bins.
* `params["angles"]` has shape `[5, 20, 140]` corresponding to the 5 angles in `cgdms.angles`, the 20 amino acids in `cgdms.aas` and 140 angle bins.
* `params["dihedrals"]` has shape `[5, 60, 142]` corresponding to the 5 dihedrals in `cgdms.dihedrals`, 20 amino acids in each predicted secondary structure type (ala helix, ala sheet, ala coil, arg helix, etc.) and 140 angle bins with an extra 2 to wrap round and allow periodicity.

## Notes

Running a simulation takes x of GPU memory.
Training a model takes up to 32 GB of GPU memory.

The code in this package is set up to run specific coarse-grained simulations of proteins.
However, the package contains code that could be useful to others wishing to carry out general differentiable simulations with PyTorch.
This includes integrators not used in the paper and not thoroughly tested (velocity-free Verlet, two Langevin implementations), the Andersen thermostat, RMSD with the Kabsch algorithm, and code to apply forces to atoms from bond angle and dihedral angle potentials.

Other software related to differentiable molecular simulation includes [Jax MD](https://github.com/google/jax-md), [TorchMD](https://github.com/torchmd), [DeePMD-kit](https://github.com/deepmodeling/deepmd-kit), [SchNetPack](https://github.com/atomistic-machine-learning/schnetpack), [DiffTaichi](https://github.com/yuanming-hu/difftaichi), [Time Machine](https://github.com/proteneer/timemachine) and [Molly](https://github.com/JuliaMolSim/Molly.jl).
