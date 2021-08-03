import sys
"""
Example of a job script containing several computational steps 
for structure optimization using the ASE framework interfacing 
to VASP.
"""

def job(kpar=1,ncore=1):
    """
    job function with arguments kpar and ncore as parallelization options
    """
    import vasplib 
    from ase.calculators.vasp import Vasp
    from ase.io import read
    import numpy as np
   
    #General job information
    job_info = {'atoms': 'atoms/CuTest',
                'atomtype': 'vasp',
                'subdir':'results',
                'name': 'CuTest'}

    #Job information            
    atoms = read(job_info['atoms'],format=job_info['atomtype'])

    #Define VASP job parameters
    calc = Vasp(setups='recommended',
                xc='PBE',           #PAW potentials [PBE or LDA]
                prec='normal',      #specifies the "precision"-mode [Accurate strictly avoids any aliasing or wrap around errors]
                ediff=1e-6,         #Convergence criterion [eV]
                nsw=0,              #maximum number of ionic steps
                ibrion=-1,          #determines how the ions are updated and moved [-1: no update, 0: molecular dynamics, 1: ionic relaxation (RMM-DIIS) ...]
                nbands=30,          #number of orbitals [default: NELECT/2+NIONS/2	non-spinpolarized]
                lcharg='.TRUE.',    #determines whether the charge densities (files CHGCAR and CHG) are written
                icharg = 1,         #determines how VASP constructs the initial charge density
                kpts=[5, 5, 5],     #specifies the Monkhorst-Pack grid
                ismear=1,           #Methfessel Paxton energy smearing order
                sigma=0.2)          #Smearing energy [eV]
    calc.set(atoms=atoms)           #Set ASE Atoms object
    calc.set(kpar=int(kpar), ncore=int(ncore)) #Set kpar and ncore

    ### WORKFLOW ###
    #ENCUT convergence
    job_info['method'] = 'encut_conv'
    param = {'name': 'encut',
             'unit': 'eV', 
             'epsilon': 0.01,
             'stop_when_converged': True,
             'value': [*range(100, 400+1, 10)]}
    [energy, calc_conv] = vasplib.param_convergence(calc,param,job_info)

    #KPOINT convergence
    job_info['method'] = 'kpoint_conv'
    param = {'name': 'kpts',
             'unit': '1', 
             'epsilon': 0.01,
             'stop_when_converged': True,
             'value': [*range(3, 15+1, 2)]}         
    param['value'] = [[p,p,p] for p in param['value']] #Make set of 3 k-points
    [energy, calc_conv] = vasplib.param_convergence(calc_conv,param,job_info)

    #VOLUME Relaxation - Equation of State
    calc_conv.set(isif=2,     # ISIF=2: Update ion positions, keep cell shape and volume
             ibrion=1)   # IBRION=1: ionic relaxation
    v0 = calc_conv.get_atoms().get_volume()
    job_info['method'] = 'ISIF2_rel'
    param = {'name': 'volume',
             'unit': 'AA^3', 
             'value': (v0*np.arange(0.8, 1.21, 0.05)).tolist()}         
    [volumes, energies, eos] = vasplib.volume_relaxation(calc,param,job_info)

    #VOLUME Refinement - Equation of State
    job_info['load_dir'] =  job_info['method']+'/raw' 
    job_info['prev_results'] = job_info['method']+'/tables/volume_vs_energy.json'
    job_info['method'] = 'ISIF4_ref'
    calc = vasplib.load_calc(job_info)

    #Set calculation parameters
    for c in calc:
        c.set(isif=4,   #ISIF=2: Update ion positions, keep cell shape and volume
              ibrion=1) # IBRION=1: ionic relaxation   

    [volumes, energies, eos] = vasplib.volume_refinement(calc,job_info)


if __name__ == '__main__':
    # Map command line arguments to function arguments.
    job(*sys.argv[1:])