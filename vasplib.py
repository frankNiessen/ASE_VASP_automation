from ase.calculators.vasp import Vasp
from ase.eos import EquationOfState
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import math

'''This is a library for autoamted routines for VASP using ASE'''
matplotlib.use('Agg')

def param_convergence(calc,param,job_info):
    '''Parameter convergence study by iterative parameter change and potential energy calculation'''
    #Initialize setting
    energy = []
    proj_dir = job_info['subdir']+'/'+job_info['name']+'/'+job_info['method']
    print('---------------------------')
    print('Running ' + param['name'] + ' convergence ...')
    if 'epsilon' in param.keys(): print('   -Tolerance: ' + str(param['epsilon']) + ' eV') 
    if 'stop_when_converged' in param.keys(): print('   -Stopping when converged') 
    print('---------------------------')

    #Loop over parameter and calculate the potential energy
    for idx, item in enumerate(param['value']):
        print(str(idx+1)+'/'+str(len(param['value']))+' - ' + param['name'] + ': ' +
              str(item) + ' (' + param['unit'] + '),',end="", flush=True)
        if all(isinstance(x, int) for x in param['value']):
            calc.set(directory=proj_dir+'/raw/{0}'.format(item))
        elif all(isinstance(x, float) for x in param['value']):
            calc.set(directory=proj_dir+'/raw/{0:1.2f}'.format(item))
        elif all(isinstance(x, list) for x in param['value']):
            outstr = format(item).replace(",","_").strip("[]").replace(" ","")
            calc.set(directory=proj_dir+'/raw/'+outstr)
        eval('calc.set('+param['name']+'='+str(item)+')')             
        atoms = calc.get_atoms()
        energy.append(atoms.get_potential_energy())

        #Check for errors in OUTCAR
        __error_check__(calc)
        
        #Print energies and check for convergence
        if 'epsilon' in param.keys() and idx > 0 and np.abs(energy[idx]-energy[idx-1])<param['epsilon']:
            print(" e_pot: {0} eV [converged]".format(energy[idx]))
            calc_conv = calc
            if 'stop_when_converged' in param.keys() and param['stop_when_converged']:
                print("Stopping convergence loop: difference in e_pot < " + str(param['epsilon']) + ' eV')
                param['value'][idx+1:] = []
                break
        else:
            print(" e_pot: {0:1.6f} eV".format(energy[idx]))

    #Save table
    __tojson__(proj_dir,param['value'],energy,param)
    #Plot and save figure
    if all(isinstance(x, list) for x in param['value']):   
        __makeplot__([i[0] for i in param['value']],energy,param)
    else:
        __makeplot__(param['value'],energy,param)
    __saveplot__(proj_dir,param['name'] + '_vs_energy.png')
    return [energy, calc_conv]


def volume_relaxation(calc,param,job_info):
    '''Volume relaxation by systematically varied cell volume and potential energy calculation'''
    energy, volume = [], []
    atoms = calc.get_atoms()
    v0 = atoms.get_volume()
    cell0 = atoms.get_cell()
    proj_dir = job_info['subdir']+'/'+job_info['name']+'/'+job_info['method']
 
    print('---------------------------')
    print('Running volume relaxation: ' + job_info['method'] + ' ...')
    print('---------------------------')
    for idx, item in enumerate(param['value']):
        print(str(idx+1)+'/'+str(len(param['value']))+' - ' + param['name'] + 
              ': {0:1.2f}'.format(item) + ' (' + param['unit'] + '),',end="", flush=True)
        cell_factor = (item / v0)**(1. / 3.)
        atoms.set_cell(cell0 * cell_factor, scale_atoms=True)
        calc.set(directory=proj_dir+'/raw/{0:1.2f}'.format(item))
        calc.set(atoms=atoms)    
        energy.append(atoms.get_potential_energy())
        __error_check__
        volume.append(atoms.get_volume())
        print(" e_pot: {0:1.6f} eV".format(energy[idx]))

    #Save table
    __tojson__(proj_dir,param['value'],energy,param)
    #Plot and save figure
    __makeplot__(param['value'],energy,param)
    __saveplot__(proj_dir,param['name'] + '_vs_energy.png')

    #Fit an equation of state
    eos = EquationOfState(volume,energy)
    v0, e0, B = eos.fit()
    print('.......... Equation of State ..........')
    print('v0 = {0} AA^3, E0 = {1} eV, B  = {2} eV/A^3'.format(v0, e0, B))
    eos.plot()
    __saveplot__(proj_dir,'equation_of_state.png')

    return [volume, energy, [v0,e0,B]]


def volume_refinement(calcs,job_info):
    '''Volume relaxation by refinement of previous calculation and potential energy calculation'''
    energy, volume = [], []
    param = {'name': 'volume',
             'unit': 'AA^3'}
    proj_dir = job_info['subdir']+'/'+job_info['name']+'/'+job_info['method']

    print('---------------------------')
    print('Running volume refinement: ' + job_info['method'] + ' ...')
    print('---------------------------')
    for idx, calc in enumerate(calcs):
        atoms = calc.get_atoms()
        print(str(idx+1)+'/'+str(len(calcs))+' - Refining job ' + job_info['load_dir']+ 
                ': volume {0:1.2f} AA^3,'.format(atoms.get_volume()) ,end="", flush=True)
        calcname = os.path.basename(calc.directory)
        calc.set(directory=proj_dir+'/'+calcname)
                
        #Calculate 
        atoms = calc.get_atoms()
        energy.append(atoms.get_potential_energy())
        __error_check__
        volume.append(atoms.get_volume())
        print(" e_pot: {0:1.6f} eV".format(energy[idx]))

    #Save table
    __tojson__(proj_dir,volume,energy,param)

    #Plot and save figure
    __makeplot__(volume,energy,param)
    __saveplot__(proj_dir,param['name'] + '_vs_energy.png')

    #Fit an equation of state
    eos = EquationOfState(volume,energy)
    v0, e0, B = eos.fit()
    print('.......... Equation of State ..........')
    print('v0 = {0} AA^3, E0 = {1} eV, B  = {2} eV/A^3'.format(v0, e0, B))
    eos.plot()
    __saveplot__(proj_dir,'equation_of_state.png')

    #Compare to previous results
    data = __fromjson__(job_info)
    if data:
        plt.figure()
        __makeplot__(volume,energy,param)
        __makeplot__(data['volume (AA^3)'],data['energies (eV)'],param)
        plt.legend(['refined', 'initial'], loc='best')
        __saveplot__(proj_dir,'refinement_results.png')
    return [volume, energy, [v0,e0,B]]

def load_calc(job_info):
    import os
    load_dir = job_info['subdir'] + '/' + job_info['name'] + '/' + job_info['load_dir']
    #Load previous calculation
    if os.path.exists(load_dir) and os.path.exists(load_dir+'/OUTCAR'):
        calc = Vasp(directory=load_dir, restart=True)
    elif os.path.exists(load_dir):
        load_dirs = sorted([x[0] for x in os.walk(load_dir)][1:])
        if all([os.path.exists(s+'/OUTCAR') for s in load_dirs]):
            calc = [Vasp(directory=v, restart=True) for v in load_dirs]
        else:
            raise Exception('"job_info[load_dir]" does not point to a VASP directory!')    
    else:
        raise Exception('Path ' + load_dir + ' does not exist!')    
    return calc


def get_valence_electrons(calc, filename=None):

    """
    Return the number of valence electrons for the atoms.
    Calculated from the POTCAR file.
    from https://github.com/jkitchin/vasp#vaspget_valence_electrons
    """

    default_electrons = get_default_number_of_electrons(calc, filename)

    d = {}
    for s, n in default_electrons:
        d[s] = n
    atoms = calc.get_atoms()

    nelectrons = 0
    for atom in atoms:
        nelectrons += d[atom.symbol]
    return int(nelectrons)


def get_default_number_of_electrons(calc, filename):
    """
    Return the default electrons for each species.
    from https://github.com/jkitchin/vasp#vaspget_default_number_of_electrons
    """
    if filename is None:
        filename = os.path.join(calc.directory, 'POTCAR')

    calc.write_input(calc.get_atoms())

    nelect = []
    lines = open(filename).readlines()
    for n, line in enumerate(lines):
        if line.find('TITEL') != -1:
            symbol = line.split('=')[1].split()[1].split('_')[0].strip()
            valence = float(lines[n + 4].split(';')[1]
                            .split('=')[1].split()[0].strip())
            nelect.append((symbol, valence))
    return nelect


def get_number_of_bands(atoms,job_info,f=1.0):
    """
    Determines the number of bands for a structure according to 
    https://www.vasp.at/wiki/index.php/Number_of_bands_NBANDS
    and an optional prefactor f
    Args:
        atoms (Atoms): ASE collection of atoms https://wiki.fysik.dtu.dk/ase/ase/atoms.html 
        job_inf (dict): Dictionary with job info on directories
        f (float): Linear prefactor to increase the number of bands
    Returns:
        (nbands): Number of bands.
    """
    calc = Vasp(directory=job_info['subdir']+'/.temp')
    calc.set(atoms=atoms)
    n_electrons = get_valence_electrons(calc)
    n_ions = atoms.get_global_number_of_atoms()
    n_bands = int((0.5*(n_electrons+n_ions))*f)
    print("The cell has {0} valence electrons and {1} ions. For f = {2:1.1f} the number of bands should be 0.5*f*(N_ELEC+N_ION) = {3}".format(n_electrons, n_ions, f, n_bands))
    return n_bands

def get_number_of_cores(n_bands,fac_bpc = 8,base = [24,40]):
    """
    Determines the number of cores based on the number of bands according to 
    https://www.nsc.liu.se/~pla/blog/2015/01/12/vasp-how-many-cores/
    based on a bands_per_core factor and base (cores per node)
    Args:
        n_bands (int): Number ob bands 
        fac_bpc (int): Bands_per_core factor
        base ([int]): Cores per node to consider
    Returns:
        Text output
    """
    n_cores = math.ceil(n_bands/fac_bpc)    
    n_cores_r = [int(b * math.ceil(float(n_cores)/b)) for b in base]
    n_nodes_r = [int(n_cores_r[i]/b) for i,b in enumerate(base)]
    mod = [n_cores % b for b in base]
    n_bands_r = [int(fac_bpc * n) for n in n_cores_r]
    for i,b in enumerate(base):
        print("............")
        print("Using {0}-core nodes the number of cores is {1} and the number of nodes {2}.".format(b, n_cores_r[i], n_nodes_r[i]))
        print("{0} extra cores were added, increasing the number of bands from {1} to {2}.".format(mod[i], n_bands, n_bands_r[i]))
       

def __error_check__(calc):
    '''Checking for errors after VASP calculation'''

    if not calc.converged:
            print("Stopping convergence loop: Calculation has not converged!")
            raise Exception('VASP did not converge')
    if [True for s in calc.load_file('OUTCAR') if 'TOO FEW BANDS!!' in s]:
            raise Exception('Too few bands! Increase the parameter NBANDS in file INCAR to ensure that the highest band is unoccupied at all k-points')


def __tojson__(proj_dir,x,energy,param):
    '''Export results to *.json file'''
    if not os.path.exists(proj_dir+'/tables'):
        os.makedirs(proj_dir+'/tables')
    with open(proj_dir+'/tables'+'/' + param['name'] + '_vs_energy.json', 'w') as f:
        f.write(json.dumps({param['name']+' ('+param['unit']+')': x, 'energies (eV)': energy}))
    

def __fromjson__(job_info):
    '''Import results from *.json file'''
    fname = job_info['subdir']+'/'+job_info['name']+'/'+job_info['prev_results']
    if os.path.exists(fname):
        f = open(fname)
        data = json.load(f)
        return data


def __makeplot__(x,energy,param):
    '''Make a simple x-y plot'''
    #Plot results
    plt.scatter(x, energy)
    plt.plot(x, energy)
    plt.xlabel(param['name']+' ('+param['unit']+')')
    plt.ylabel('Total Energy (eV)') 

def __saveplot__(proj_dir,plotname):
    '''Save plot'''
    #Save plotted results
    if not os.path.exists(proj_dir+'/images'):
        os.makedirs(proj_dir+'/images')
    plt.savefig(proj_dir+'/images'+'/'+plotname)
