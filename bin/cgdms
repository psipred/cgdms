#!/usr/bin/env python

# Argument handling
# Author: Joe G Greener

import argparse
import pkg_resources

parser = argparse.ArgumentParser(description=(
    "Differentiable molecular simulation of proteins with a coarse-grained potential. "
    "See https://github.com/psipred/cgdms for documentation and citation information. "
    f"This is version {pkg_resources.get_distribution('cgdms').version} of the software."
))
subparsers = parser.add_subparsers(dest="mode",
    help="the mode to run cgdms in, run \"cgdms {mode} -h\" to see help for each")

parser_makeinput = subparsers.add_parser("makeinput",
    description="Form an input protein data file.", help="form an input protein data file")
parser_makeinput.add_argument("-i", "--input", required=True,
    help="PDB/mmCIF file, format guessed from extension")
parser_makeinput.add_argument("-s", "--ss2",
    help="PSIPRED ss2 file, default fully coiled (not recommended)")

parser_simulate = subparsers.add_parser("simulate",
    description="Run a coarse-grained simulation of a protein in the learned potential.",
    help="run a coarse-grained simulation of a protein in the learned potential")
parser_simulate.add_argument("-i", "--input", required=True, help="input protein data file")
parser_simulate.add_argument("-o", "--output",
    help="output PDB filepath, if this is not given then no PDB file is written")
parser_simulate.add_argument("-s", "--startconf", required=True,
    choices=["native", "extended", "predss", "random", "helix"], help="starting conformation")
parser_simulate.add_argument("-n", "--nsteps", required=True, type=float,
    help="number of simulation steps to run")
parser_simulate.add_argument("-t", "--temperature", type=float, default=0.015,
    help="thermostat temperature, default 0.015")
parser_simulate.add_argument("-c", "--coupling", type=float, default=25.0,
    help="thermostat coupling constant, default 25, set to 0 to run without a thermostat")
parser_simulate.add_argument("-st", "--starttemperature", type=float, default=0.015,
    help="starting temperature, default 0.015")
parser_simulate.add_argument("-ts", "--timestep", type=float, default=0.004,
    help="integrator time step, default 0.004")
parser_simulate.add_argument("-r", "--report", type=int, default=10_000,
    help="interval for printing energy and writing PDB file, default 10_000")
parser_simulate.add_argument("-p", "--parameters",
    help="parameter set to use, default is the trained model")
parser_simulate.add_argument("-d", "--device",
    help="device to run on, default is \"cuda\" if it is available otherwise \"cpu\"")
parser_simulate.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], default=2,
    help="logging verbosity, default 2")

parser_energy = subparsers.add_parser("energy",
    description="Calculate the energy of a structure in the learned potential.",
    help="calculate the energy of a structure in the learned potential")
parser_energy.add_argument("-i", "--input", required=True, help="input protein data file")
parser_energy.add_argument("-m", "--minsteps", type=float, default=0.0,
    help="number of minimisation steps to run, default 0")
parser_energy.add_argument("-p", "--parameters",
    help="parameter set to use, default is the trained model")
parser_energy.add_argument("-d", "--device",
    help="device to run on, default is \"cuda\" if it is available otherwise \"cpu\"")
parser_energy.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], default=0,
    help="logging verbosity, default 0")

parser_thread = subparsers.add_parser("thread",
    description=("Calculate the energy in the learned potential of a set of sequences threaded "
                    "onto a structure. Returns one energy per line."),
    help=("calculate the energy in the learned potential of a set of sequences threaded onto a "
            "structure"))
parser_thread.add_argument("-i", "--input", required=True, help="input protein data file")
parser_thread.add_argument("-s", "--sequences", required=True,
    help="file of sequences to thread, one per line")
parser_thread.add_argument("-m", "--minsteps", type=float, default=100.0,
    help="number of minimisation steps to run for each sequence, default 100")
parser_thread.add_argument("-p", "--parameters",
    help="parameter set to use, default is the trained model")
parser_thread.add_argument("-d", "--device",
    help="device to run on, default is \"cuda\" if it is available otherwise \"cpu\"")
parser_thread.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], default=0,
    help="logging verbosity, default 0")

parser_train = subparsers.add_parser("train",
    description="Train the model. This can take a couple of months... go and get a coffee?",
    help="train the model")
parser_train.add_argument("-o", "--output", default="cgdms_params.pt",
    help="output learned parameter filepath, default \"cgdms_params.pt\"")
parser_train.add_argument("-d", "--device",
    help="device to run on, default is \"cuda\" if it is available otherwise \"cpu\"")
parser_train.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], default=0,
    help="logging verbosity, default 0")

args = parser.parse_args()

for arg in ("nsteps", "temperature", "coupling", "starttemperature",
            "timestep", "report", "minsteps"):
    if arg in args.__dict__ and args.__dict__[arg] < 0:
        raise argparse.ArgumentTypeError(f"{arg} is {args.__dict__[arg]} but must be non-negative")

# Imported after argument parsing because it takes a few seconds
from cgdms import *

if "device" in args.__dict__:
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

if args.mode == "makeinput":
    print_input_file(args.input, args.ss2)
elif args.mode == "simulate":
    if args.output and os.path.exists(args.output):
        os.remove(args.output)
    if args.parameters:
        params = torch.load(args.parameters, map_location=device)
    else:
        params = torch.load(trained_model_file, map_location=device)
    native_coords, inters_flat, inters_ang, inters_dih, masses, seq = read_input_file(args.input,
                                                                                    device=device)
    with torch.no_grad():
        simulator = Simulator(params["distances"], params["angles"], params["dihedrals"])
        if args.startconf == "native":
            coords = native_coords
        else:
            coords = starting_coords(seq, conformation=args.startconf, input_file=args.input,
                                        device=device)
        coords = simulator(coords.unsqueeze(0), inters_flat.unsqueeze(0), inters_ang.unsqueeze(0),
                            inters_dih.unsqueeze(0), masses.unsqueeze(0), seq,
                            native_coords.unsqueeze(0), int(args.nsteps), integrator="vel",
                            timestep=args.timestep, start_temperature=args.starttemperature,
                            thermostat_const=args.coupling, temperature=args.temperature,
                            sim_filepath=args.output, report_n=args.report,
                            verbosity=args.verbosity)
elif args.mode == "energy":
    if args.parameters:
        params = torch.load(args.parameters, map_location=device)
    else:
        params = torch.load(trained_model_file, map_location=device)
    coords, inters_flat, inters_ang, inters_dih, masses, seq = read_input_file(args.input,
                                                                                device=device)
    with torch.no_grad():
        simulator = Simulator(params["distances"], params["angles"], params["dihedrals"])
        energy = simulator(coords.unsqueeze(0), inters_flat.unsqueeze(0), inters_ang.unsqueeze(0),
                            inters_dih.unsqueeze(0), masses.unsqueeze(0), seq, coords.unsqueeze(0),
                            int(args.minsteps), integrator="min", energy=True,
                            verbosity=args.verbosity)
        print(round(energy.item(), 3))
elif args.mode == "thread":
    if args.parameters:
        params = torch.load(args.parameters, map_location=device)
    else:
        params = torch.load(trained_model_file, map_location=device)
    with torch.no_grad():
        simulator = Simulator(params["distances"], params["angles"], params["dihedrals"])
        with open(args.sequences) as f:
            for li, line in enumerate(f):
                if not line.startswith(">"):
                    coords, inters_flat, inters_ang, inters_dih, masses, seq = read_input_file_threaded(
                                                        args.input, line.rstrip(), device=device)
                    energy = simulator(coords.unsqueeze(0), inters_flat.unsqueeze(0),
                                        inters_ang.unsqueeze(0), inters_dih.unsqueeze(0),
                                        masses.unsqueeze(0), seq, coords.unsqueeze(0),
                                        int(args.minsteps), integrator="min", energy=True,
                                        verbosity=args.verbosity)
                    print(li + 1, round(energy.item(), 3))
elif args.mode == "train":
    train(args.output, device=device, verbosity=args.verbosity)
else:
    print("No mode selected, run \"cgdms -h\" to see help")
