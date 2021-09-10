import sys

def job(kpar=1,ncore=1):
    import vasplib 
    from ase.calculators.vasp import Vasp
    from ase.io import read
    import numpy as np
    from ase.lattice.hexagonal import HexagonalClosedPacked
   
    #General job information
    job_info = {'atoms': 'atoms/Fe_hcp',
                'atomtype': 'vasp',
                'subdir':'results',
                'name': 'Fe_hcp'}
                

    atoms = read(job_info['atoms'],format=job_info['atomtype'])

    #Define VASp job parameters
    calc = Vasp(setups='recommended',
                xc='PBE',           #PAW potentials [PBE or LDA]
                prec='normal',      #specifies the "precision"-mode [Accurate strictly avoids any aliasing or wrap around errors]
                ediff=1e-6,         #Convergence criterion [eV]
                nsw=0,              #maximum number of ionic steps
                ibrion=-1,          #determines how the ions are updated and moved [-1: no update, 0: molecular dynamics, 1: ionic relaxation (RMM-DIIS) ...]
                nbands=40,          #number of orbitals [default: NELECT/2+NIONS/2	non-spinpolarized]
                lcharg='.TRUE.',    #determines whether the charge densities (files CHGCAR and CHG) are written
                icharg = 1,         #determines how VASP constructs the initial charge density
                kpts=[5, 5, 5],     #specifies the Monkhorst-Pack grid
                ismear=1,           #Methfessel Paxton energy smearing order
                sigma=0.2)          #Smearing energy [eV]
    calc.set(atoms=atoms)
    calc.set(kpar=int(kpar), ncore=int(ncore))

    #ENCUT convergence
    job_info['method'] = 'encut_conv'
    param = {'name': 'encut',
             'unit': 'eV', 
             'epsilon': 0.01,
             'stop_when_converged': True,
             'value': [*range(200, 400+1, 10)]}
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
    atoms = calc_conv.get_atoms()
    abc = atoms.get_cell_lengths_and_angles()
    a_range = np.arange(0.96,1.04,0.01)*abc[1]
    ca_range = np.arange(0.96,1.04,0.01)*abc[2]/abc[1]

    v0 = calc_conv.get_atoms().get_volume()
    job_info['method'] = 'ISIF2_rel'
    param = {'name': ('a', 'c/a'),
             'unit': ('AA', '1'), 
             'value': [a_range,ca_range]}         
    [volumes, energies, eos] = vasplib.volume_relaxation_hcp(calc,param,job_info)

    #VOLUME Refinement - Equation of State
    job_info['load_dir'] =  job_info['method']+'/raw' 
    job_info['prev_results'] = job_info['method']+'/tables/volume_vs_energy.json'
    job_info['method'] = 'ISIF4_ref'
    load_dir = job_info['subdir'] + '/' + job_info['name'] + '/' + job_info['load_dir']
    calc = vasplib.load_calc(load_dir)

    #Set calculation parameters
    for c in calc:
        c.set(isif=4,   #ISIF=2: Update ion positions, keep cell shape and volume
              ibrion=1) # IBRION=1: ionic relaxation   

    [volumes, energies, eos] = vasplib.volume_refinement(calc,job_info)


if __name__ == '__main__':
    # Map command line arguments to function arguments.
    job(*sys.argv[1:])
