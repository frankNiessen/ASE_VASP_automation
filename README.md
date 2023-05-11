# DFT automatization using VASP and ASE
This repo contains a library that aims at automatizing some Density Functional Theory (DFT) workflows in [**VASP**](https://vasp.at/) by using the [**ASE**](https://wiki.fysik.dtu.dk/ase/index.html) toolkit. The workflows are  losely based on the wonderful ressources provided by [Prof. John Kitchin](http://kitchingroup.cheme.cmu.edu/dft-book/dft.html). 
I only implement what I need for my own research and I share it here in the hope that it might benefit others.

*This repo has been entirely revamped on 11/05/2023. You can find the legacy version of ASE_VASP_automation [here](https://github.com/frankNiessen/ASE_VASP_automation/releases/tag/v0.9).*

## Installation
- You need to have a properly working and licensed **VASP** installation on your local computer or a high performance cluster.
- You need to have **ASE** properly installed and set up to work with VASP (check out the instructions [here](https://wiki.fysik.dtu.dk/ase/ase/calculators/vasp.html)). The workflow has been tested with version *3.20.1*.
- To install the ASE_VASP_automation software, paste this folder wherever it is convenient for you on your computer and make sure that it is indexed in the [PYTHONPATH environment variable](https://www.tutorialspoint.com/What-is-PYTHONPATH-environment-variable-in-Python).
- The scripts can be run from anywhere, meaning that you could cut and paste the example folder to wherever you store your code.

## Program structure
The program is build to run simple tasks that are defined within *.json* files. Within the example folder, few scripts are provided in which workflows are defined. Workflows comprise a certain order of tasks that are conducted on defined atomic structures given a set of parameters. The folder *settings* contains settings files that can be used in conjunction with different workflows.

## Examples
The currently only provided example is on stacking fault energy determination using the axial interaction model (see our [recent paper](https://github.com/frankNiessen/ASE_VASP_automation/blob/master/examples/SFE_AIM_Fe/Niessen_Li_Werner_Lu_Vitos_Villa_Somers_2023.pdf) on its application to Fe-C, Fe-N and austenitic stainless steel). Here 4 different relaxation modes are offered: volume relaxation, relaxation of the c/a ratio, volume relaxation and relaxation of the c/a ratio, and no relaxation at all. Finally, the SFE value is determined using the axial interaction model.

## Parallelization
The scripts can be run as-is on a personal computer, by default it will do so with *kpar = 1* and *ncore = 1*. If you have **VASP** and **ASE** installed on a High Performance Cluster (*HPC*), or if you have a parallelized version of **VASP** compiled on your personal computer, you can supply the arguments *kpar* and *ncore* on the command line. As an example, you can run the calculations with *kpar = 2* and *ncore = 16* by executing: 
*job.py 2 16*

If you want to run **VASP** in a parallel environment, you will have to set the **ASE** environment variable *ASE_VASP_COMMAND* to *"mpirun vasp_std"*.

## Contribute
If you find any errors, feel free to submit a pull request or to adress them in the "Issues" tab. If you want to contribute to this code, please get into contact with me. Also feel tree to fork the code and develop your own alterations.
