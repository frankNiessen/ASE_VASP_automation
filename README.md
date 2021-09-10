# DFT automatization using VASP and ASE
This repo contains a library that aims at automatizing some Density Functional Theory (DFT) workflows in [**VASP**](https://vasp.at/) by using the [**ASE**](https://wiki.fysik.dtu.dk/ase/index.html) toolkit. The workflows are adapted from the wonderful ressources provided by [Prof. John Kitchin](http://kitchingroup.cheme.cmu.edu/dft-book/dft.html). I am not an expert in DFT but I hope that these scripts might be helpful to some people starting up with DFT and management of DFT workflows.

## Installation
- You need to have a properly working and licensed **VASP** installation on your local computer or a high performance cluster.
- You need to have **ASE** properly installed and set up to work with VASP (check out the instructions [here](https://wiki.fysik.dtu.dk/ase/ase/calculators/vasp.html)). The workflow has been tested with version *3.20.1*.

## Structure
The repo consists of two scripts and a function library. The script *create_atoms.py* creates a structure and determines the number of bands and cores that should be allocated for a calculation. The script *job.py* describes the details of a job, which is a workflow consisting of several consecutive minimization procedures. In the example, some convergence tests and minimization procedures are performed on a Copper structure.

## Scripts
### create_atoms.py
This script is used to create/load a structure and determine the number of bands and cores for a calculation based on some common formulae. This is useful to estimate the parallelization options when optimizing large super-cells. Created structures are saved to and loaded from the subfolder *ạtoms*.

### job.py
The job script shows an example workflow on a Copper structure. Here the parameters *e-cut* and *kpoints* are determined by convergence testing. Then the structure is optimized with different conditions for structure relaxation. All results are saved in a subfolder *results* which is created upon the first execution of the program. The results contain the raw input and output files from **VASP** as well as the results from convergence and optimization runs in the form of ASCII files and images of the plots.

## Parallelization
The scripts can be run as-is on a personal computer, by default it will do so with *kpar = 1* and *ncore = 1*. If you have **VASP** and **ASE** installed on a High Performance Cluster (*HPC*), or if you have a parallelized version of **VASP** compiled on your personal computer, you can supply the arguments *kpar* and *ncore* on the command line. As an example, you can run the calculations with *kpar = 2* and *ncore = 16* by executing: 
*job.py 2 16*

If you want to run **VASP** in a parallel environment, you will have to set the **ASE** environment variable *ASE_VASP_COMMAND* to *"mpirun vasp_std"*.

## Contribute
If you find any errors, feel free to submit a pull request or to adress them in the "Issues" tab. If you want to contribute to this code, please get into contact with me. Also feel tree to fork the code and develop your own alterations.
