from ase.calculators.calculator import equal
from ase.calculators.vasp import Vasp
from ase.eos import EquationOfState
from ase.io import read
from ase import lattice
from datetime import datetime
import time
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import math
import sys

'''This is a library for autoamted routines for VASP using ASE'''
matplotlib.use('Agg')


def param_convergence(calc, task):
    '''Parameter convergence study by iterative parameter change and potential energy calculation'''
    prnt_subheader('Running ' + task['name'])
    print(
        '   -Tolerance: {0:0.2e}'.format(task['epsilon_value']) + task['epsilon_unit'])
    if 'stop_when_converged' in task.keys() and task['stop_when_converged']:
        print('   -Stopping when converged')

    # Prepare variables
    values = task['value']
    outdir = task['outdir']
    result, unit = __result_template__(task['param'], task['unit'])

    # Loop over parameter and calculate the total energy
    for idx, value in enumerate(values):
        print(str(idx + 1) + '/' + str(len(values)) + ' - ' + task['name'] + ': ' +
              str(value) + ' (' + task['unit'] + '),', end="", flush=True)
        if all(isinstance(x, int) for x in values):
            calcdir = outdir + '/raw/{0}'.format(value)
        elif all(isinstance(x, float) for x in task['value']):
            calcdir = outdir + '/raw/{0:1.2f}'.format(value)
        elif all(isinstance(x, list) for x in task['value']):
            outstr = format(value).replace(
                ",", "_").strip("[]").replace(" ", "")
            calcdir = outdir + '/raw/' + outstr
        calc.set(directory=calcdir)
        eval('calc.set(' + task['param'] + '=' + str(value) + ')')

        calc, e, v = calculate(calc)
        result["etot"].append(e)
        result["v0"].append(v)
        result[task["param"]].append(value)

        # Print energies and check for convergence
        if 'epsilon_value' in task.keys() and idx > 0 and np.abs(result["etot"][idx] - result["etot"][idx - 1]) < task['epsilon_value']:
            if 'stop_when_converged' in task.keys() and task['stop_when_converged']:
                print("Stopping convergence loop: Difference in e_pot < {0:0.2e} {1}".format(
                    task['epsilon_value'], task['epsilon_unit']))
                task['value'][idx + 1:] = []
                break

    return calc, result, unit


def kpoint_optimization(calc, task):
    task['value'] = kpoint_lists(calc, task['value'])
    return param_convergence(calc, task)


def cell_relaxation(calc, task):
    '''Cell relaxation by systematically varied cell volume and potential energy calculation'''
    atoms = calc.get_atoms()
    v0 = atoms.get_volume()
    cell0 = atoms.get_cell()

    # Prepare variables
    values = task['value']
    outdir = task['outdir']
    result, unit = __result_template__(task['param'], task['unit'])

    prnt_subheader('Running ' + task['name'])
    for idx, value in enumerate(task['value']):
        print(str(idx + 1) + '/' + str(len(task['value'])) + ' - ' + task['name'] +
              ': {0:1.2f}'.format(value) + ' (' + task['unit'] + '),', end="", flush=True)
        atoms = stretch_cell(atoms, cell0, value)
        calc.set(atoms=atoms)

        calc_dir = outdir + '/raw/{0:1.2f}'.format(value)
        calc.set(directory=calc_dir)
        calc, e, v = calculate(calc)
        result["etot"].append(e)
        result["v0"].append(v)
        result[task["param"]].append(value)

    # Reset atoms to cell0
    atoms = stretch_cell(atoms, cell0, 1)
    calc.set(atoms=atoms)
    return calc, result, unit


def fit_eos(calc, task):
    # Fit an equation of state

    # Find and open data
    filename = os.path.abspath(os.path.join(
        task["outdir"], os.pardir)) + "/" + task["input_dataset"] + "/results/" + task["job_id"] + ".json"
    with open(filename, 'r') as j:
        data = json.loads(j.read())

    # Define variables for eos fit
    x = data["results"][task["x"]]
    y = data["results"][task["y"]]
    result, unit = __result_template__("B", "eV/A^3")

    # eos fit
    eos = EquationOfState(x, y)
    x, y, dxy = eos.fit()
    result[task["x"]] = [x]
    result[task["y"]] = [y]
    result["B"] = [dxy]
    # eos command window output
    print("")
    print('.......... Equation of State ..........')
    print('{0}_0 = {1:0.4e} ({2})'.format(
        task["x"], result[task["x"]][0], data["units"][task["x"]]))
    print('{0}_0 = {1:0.4e} ({2})'.format(
        task["y"], result[task["y"]][0], data["units"][task["y"]]))
    print('{0}_0 = {1:0.4e} ({2})'.format("B", result["B"][0], unit["B"]))
    print('.......................................')

    # eos plotting
    eos.plot()
    saveplot(task["outdir"], task["job_id"] + "_" + task['name'] + '.png')
    return calc, result, unit


def cell_relaxation_hcp(calc, task):
    '''Cell relaxation by systematically varied c/a ratio of hcp cell'''
    atoms0 = calc.get_atoms()
    v0 = atoms0.get_volume()
    cell0 = atoms0.get_cell().copy()
    a0 = cell0[0, 0]
    # Prepare variables
    values = task['value']
    outdir = task['outdir']
    result, unit = __result_template__(task['param'], task['unit'])

    if not task['method'] == 'constantV':
        cell0 = None

    prnt_subheader('Running ' + task['name'] + ' - ' + task['method'])
    for id, value in enumerate(values):
        atoms = stretch_cell(atoms0, cell0, [a0, value], cell0)
        calc.set(atoms=atoms)
        a_plot = atoms.cell.array[0, 0]  # Update a
        print('{0:d}/{1:d}: '.format(id + 1, len(values))
              + task['name'] +
              ': {0:1.4f} [{1}], {2:1.4f} [{3}]'.format(
                  a_plot, "AA", value, task['unit']),
              end="", flush=True)
        calc_dir = outdir + \
            '/raw/{0:d}-{1:d}_{2:1.4f}--{3:1.4f}'.format(
                id + 1, len(values), a_plot, value)
        calc.set(directory=calc_dir)
        calc, e, v = calculate(calc)
        result["etot"].append(e)
        result["v0"].append(v)
        result[task["param"]].append(value)
    return calc, result, unit


def cell_refinement(calcs, job_info):
    '''Cell relaxation by refinement of previous calculation and potential energy calculation'''
    energy, volume = [], []
    param = {'name': 'volume',
             'unit': 'AA^3'}
    proj_dir = job_info['subdir'] + '/' + \
        job_info['name'] + '/' + job_info['method']

    prnt_subheader('Running cell refinement: ' + job_info['method'] + ' ...')
    for idx, calc in enumerate(calcs):
        atoms = calc.get_atoms()
        print(str(idx + 1) + '/' + str(len(calcs)) + ' - Refining job ' + job_info['load_dir'] +
              ': volume {0:1.2f} AA^3,'.format(atoms.get_volume()), end="", flush=True)
        calcname = os.path.basename(calc.directory)
        calc_dir = proj_dir + '/' + calcname
        calc.set(directory=calc_dir)
        calc.set(atoms=atoms)
        calc, e, v = calculate(calc)
        energy.append(e)
        volume.append(v)

    # Save table
    result2json(proj_dir, volume, energy, param)

    # Plot and save figure
    makeplot(volume, energy, param)
    saveplot(proj_dir, param['name'] + '_vs_energy.png')

    # Fit an equation of state
    eos = EquationOfState(volume, energy)
    v0, e0, B = eos.fit()
    print('.......... Equation of State ..........')
    print('v0 = {0} AA^3, E0 = {1} eV, B  = {2} eV/A^3'.format(v0, e0, B))
    eos.plot()
    saveplot(proj_dir, 'equation_of_state.png')

    # Fit an equation of state
    if len(energy) > 1:
        eos = EquationOfState(volume, energy)
        v0, e0, B = eos.fit()
        print('.......... Equation of State ..........')
        print('v0 = {0} AA^3, E0 = {1} eV, B  = {2} eV/A^3'.format(v0, e0, B))
        eos.plot()
        saveplot(proj_dir, 'equation_of_state.png')

        # Compare to previous results
        data = __fromjson__(job_info)
        if data:
            plt.figure()
            makeplot(volume, energy, param)
            makeplot(data['volume (AA^3)'], data['energies (eV)'], param)
            plt.legend(['refined', 'initial'], loc='best')
            saveplot(proj_dir, 'refinement_results.png')
        return [volume, energy, [v0, e0, B]]
    else:
        print('v0 = {0} AA^3, E0 = {1} eV'.format(volume, energy))
        return [volume, energy, []]

    return [volume, energy, [v0, e0, B]]


def stretch_cell(atoms, cell0, param, cell_ref=None):
    if type(param) == float:
        cell_factor = param**(1. / 3.)
        atoms.set_cell(cell0 * cell_factor, scale_atoms=True)
    elif type(param) == list:
        lat = cell0.get_bravais_lattice()
        if type(lat) == lattice.HEX:  # hexagonal
            if not (len(param) == 2):
                raise Exception(
                    'a in Angstrom and the c/a ratios are required for a hexagonal cell')
            if cell_ref:
                a_old = cell_ref.array[0, 0]
                coa_old = cell_ref.array[2, 2] / a_old
                coa_new = param[1]
                a_new = param[0] * (coa_old / coa_new)**(1 / 3)
                atoms.set_cell([a_new, a_new, a_new * param[1],
                                cell0.angles()[0], cell0.angles()[1], cell0.angles()[2]], scale_atoms=True)
            else:
                atoms.set_cell([param[0], param[0], param[0] * param[1],
                                cell0.angles()[0], cell0.angles()[1], cell0.angles()[2]], scale_atoms=True)

    return atoms


def kpoint_lists(calc, kpoints):
    lattice_vectors = [np.linalg.norm(c) for c in calc.get_atoms().cell]
    ratios = [(1 / l) / (1 / np.min(lattice_vectors)) for l in lattice_vectors]
    kpoints = [[k * ratios[0], k * ratios[1], k * ratios[2]]
               for k in kpoints]  # Make set of 3 k-points
    return [[round(k, 0) for k in ks]
            for ks in kpoints]  # round


def is_equal_calc(calc, calc_loaded, ignore_params=['kpar', 'ncore']):
    from deepdiff import DeepDiff
    from ase.io import read

    # Check if loaded calculation exists
    if calc_loaded == []:
        return False

    # Check for initial POSCAR file
    if not os.path.isfile(calc_loaded.directory + '/POSCAR0'):
        return False
    else:
        atoms_loaded = read(calc_loaded.directory + '/POSCAR0', format='vasp')

    # Check if the initial structures are identical
    equal_atoms = (calc.atoms.positions - atoms_loaded.positions < 1e-16).all()
    equal_cells = (calc.atoms.cell == atoms_loaded.cell).all()
    equal_elements = (calc.atoms.symbols == atoms_loaded.symbols).all()
    if not (equal_atoms and equal_cells and equal_elements):
        return False

    # Check differences in the calculation parameters
    diff = DeepDiff(calc.todict(), calc_loaded.todict())
    if not 'values_changed' in diff:
        return True
    for iparam in ignore_params:
        if "root['" + iparam + "']" in diff['values_changed']:
            del diff['values_changed']["root['" + iparam + "']"]
    is_equal = not bool(diff['values_changed'])
    return is_equal


def calculate(calc):
    calc_load = load_calc(calc.directory)
    if is_equal_calc(calc, calc_load) and calc_load.converged:
        energy = calc_load.get_potential_energy()
        volume = calc_load.get_atoms().get_volume()
    else:
        atoms_ini = calc.atoms.copy()
        energy = calc.get_potential_energy()
        __error_check__
        volume = calc.atoms.get_volume()
        atoms_ini.write(calc.directory + '/POSCAR0', 'vasp')
        calc.write_json("calc_relaxed.json")
    print(" e_pot: {0:1.6f} eV, v: {1:1.6f} AA^3".format(energy, volume))
    return calc, energy, volume


def load_calc(load_dir, raise_error=False):
    import os
    # Load previous calculation
    if os.path.exists(load_dir) and os.path.exists(load_dir + '/OUTCAR'):
        try:
            calc = Vasp(directory=load_dir, restart=True)
        except:
            return []
    elif os.path.exists(load_dir):
        load_dirs = sorted([x[0] for x in os.walk(load_dir)][1:])
        if all([os.path.exists(s + '/OUTCAR') for s in load_dirs]):
            calc = [Vasp(directory=v, restart=True) for v in load_dirs]
        else:
            if raise_error:
                raise Exception(
                    '"job_info[load_dir]" does not point to a VASP directory!')
            else:
                calc = []
    else:
        if raise_error:
            raise Exception('Path ' + load_dir + ' does not exist!')
        else:
            calc = []
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


def get_number_of_bands(atoms, job_info, f=1.0):
    """        with open(proj_dir+'/tables'+'/' + param['name'][0].replace('/','_over_') 
                + '_' + param['name'][1].replace('/','-over-') + '_vs_energy.json', 'w') as f:
            f.write(json.dumps({param['name'][0]+' ('+param['unit'][0]+')': x[0].flatten(order='F').tolist(), 
                                param['name'][1]+' ('+param['unit'][1]+')': x[1].flatten(order='F').tolist(),
                                'energies (eV)': energy.flatten().tolist()}))
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
    calc = Vasp(directory=job_info['subdir'] + '/.temp')
    calc.set(atoms=atoms)
    n_electrons = get_valence_electrons(calc)
    n_ions = atoms.get_global_number_of_atoms()
    n_bands = int((0.5 * (n_electrons + n_ions)) * f)
    print("The cell has {0} valence electrons and {1} ions. For f = {2:1.1f} the number of bands should be 0.5*f*(N_ELEC+N_ION) = {3}".format(
        n_electrons, n_ions, f, n_bands))
    return n_bands


def get_number_of_cores(n_bands, fac_bpc=8, base=[24, 40]):
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
    n_cores = math.ceil(n_bands / fac_bpc)
    n_cores_r = [int(b * math.ceil(float(n_cores) / b)) for b in base]
    n_nodes_r = [int(n_cores_r[i] / b) for i, b in enumerate(base)]
    mod = [n_cores % b for b in base]
    n_bands_r = [int(fac_bpc * n) for n in n_cores_r]
    for i, b in enumerate(base):
        print("............")
        print("Using {0}-core nodes the number of cores is {1} and the number of nodes {2}.".format(
            b, n_cores_r[i], n_nodes_r[i]))
        print("{0} extra cores were added, increasing the number of bands from {1} to {2}.".format(
            mod[i], n_bands, n_bands_r[i]))


def get_par_settings(argv):
    # Try mapping command line arguments to function argument
    if len(sys.argv[1:]) == 2:
        ncore = int(sys.argv[1])
        kpar = int(sys.argv[2])
    elif not len(sys.argv[1:]) and os.cpu_count() / 2 == 12:  # MEK-425-026-01
        ncore = 3
        kpar = 4
    else:
        ncore = 1
        kpar = 1
    print(" *Running with npar={0:d} and kpar={1:d}\n".format(ncore, kpar))
    return {"ncore": ncore, "kpar": kpar}


def __error_check__(calc):
    '''Checking for errors after VASP calculation'''

    if not calc.converged:
        print("Stopping convergence loop: Calculation has not converged!")
        raise Exception('VASP did not converge')
    if [True for s in calc.load_file('OUTCAR') if 'TOO FEW BANDS!!' in s]:
        raise Exception(
            'Too few bands! Increase the parameter NBANDS in file INCAR to ensure that the highest band is unoccupied at all k-points')


def sfe_aim(task):
    # SFE calculation according to Axial Interaction Model
    etot = []
    area = []
    for id, atoms in enumerate(task['atoms']):
        nr_at = read(task['atoms_dir'] + '/' + atoms,
                     format=task['software']).get_global_number_of_atoms()
        # Find and open data
        filename = os.path.abspath(task["result_dir"]) + "/" + atoms + "/" + \
            task["etot_source"][id] + "/results/" + \
            task["etot_job"][id] + ".json"
        with open(filename, 'r') as j:
            data = json.loads(j.read())
        etot.append(float(data["results"]["etot"]
                    [task["etot_index"][id]]) / nr_at)
        area.append(np.sqrt(3) / 2 * (np.sqrt(2) / 2 *
                    float(data["settings"]["lattice_param"]))**2 * 1e-20 * nr_at)
    result = {"sfe": [2 * (etot[1] - etot[0]) * 1.602177e-16 / (area[0])]}
    unit = {"sfe": "mJ.m^(-2)"}
    print("The SFE from the AIM approach is {0:0.2f} {1}".format(
        result["sfe"][0], unit["sfe"]))
    return result, unit


def result2json(task, settings, results, unit):
    '''Export results to *.json file'''
    time = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    outdir = task["outdir"] + '/results'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with open(outdir + '/' + task["job_id"] + '.json', 'w') as f:
        f.write(json.dumps({"info": {"job_id": task["job_id"], "time": time},
                "task": task, "settings": settings, "results": results, "units": unit}))


def __fromjson__(job_info):
    '''Import results from *.json file'''
    fname = job_info['subdir'] + '/' + \
        job_info['name'] + '/' + job_info['prev_results']
    if os.path.exists(fname):
        f = open(fname)
        data = json.load(f)
        return data


def __result_template__(param_name=None, param_unit=None):
    values = {"etot": [], "v0": []}
    units = {"etot": "eV", "v0": "AA^3"}
    if param_name:
        values[param_name] = []
    if param_unit:
        units[param_name] = param_unit
    return values, units

def ini_vasp_calculation(json_string,atoms = None, additional_settings = None, txt = "vasp.out"):
    calc = Vasp()
    calc.read_json(json_string + ".json")
    if atoms: calc.set(atoms=atoms)
    if additional_settings:
        calc.fromdict({"inputs":additional_settings})
    calc.txt = txt
    return calc

def makeplot(x, energy, param_name, param_unit):
    '''Make a simple x-y plot'''
    # Plot results
    fig, ax = plt.subplots()
    if all([type(xx) is list for xx in x]) and all([xxx == xx[0] for xx in x for xxx in xx]):
        x = [xx[0] for xx in x]
    elif all([type(xx) is list for xx in x]):
        x = [np.prod(xx) for xx in x]
    ax.scatter(x, energy)
    ax.plot(x, energy)
    ax.set_xlabel(param_name + ' (' + param_unit + ')')
    ax.set_ylabel('Total Energy (eV)')


def saveplot(outdir, filename):
    '''Save plot'''
    # Save plotted results
    if not os.path.exists(outdir + '/images'):
        os.makedirs(outdir + '/images')
    plt.savefig(outdir + '/images' + '/' + filename)
    plt.close()


def prnt_header(string):
    print('---------------------------')
    print(string)
    print('---------------------------')


def prnt_subheader(string):
    print('............................')
    print(string)
    print('............................')


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def make_jobid(uid):
    epoch = time.time()
    return "%s_%d" % (uid, epoch)


def libdir(string):
    return os.path.dirname(__file__) + string
