import sys
import json
import os
import shutil
import numpy as np
from ase.io import read
from ase import Atoms, lattice
import numpy as np
import vasplib


def execute_workflow(workflow):
    # Update run options

    workflow['run_options']['result_dir'] = workflow['run_options']['job_dir'] + '/' + workflow['run_options']['result_dir'] + \
        '/' + workflow["label"]
    workflow['run_options']["job_id"] = vasplib.make_jobid(0)
    # Delete previous calculation for "run_from_scratch" = 0
    if os.path.isdir(workflow['run_options']['result_dir']) and len(os.listdir(workflow['run_options']['result_dir'])) and workflow["run_options"]["run_from_scratch"]:
        shutil.rmtree(workflow['run_options']
                      ['result_dir'], ignore_errors=True)

    # Start logging of command line output
    sys.stdout = vasplib.Logger(
        workflow['run_options']["result_dir"] + "/" + workflow['run_options']["job_id"] + "_log.output")
    print(
        ' /////  {0} - {1}  /////'.format(workflow['name'], workflow['run_options']["job_id"]))

    # Time execution
    runtime = vasplib.runtime()

    # Start Workflow
    for ii, task in enumerate(workflow['tasks']):
        vasplib.prnt_header(
            'Running Task: {0}'.format(task))
        if workflow['tasks'][task]['method'] == 'total_energy_calculation':
            # Get lattice parameter from previous task
            if not workflow['tasks'][task]['lattice_param']:
                op_optim = workflow['tasks'][task_prev]["v_optim"]["operation"]
                op_index = workflow['tasks'][task_prev]["v_optim"]["index"]
                workflow['tasks'][task]['lattice_param'] = Atoms.fromdict(
                    workflow['tasks'][task_prev]["operations"][op_optim]["result"]["value"]["atoms"][op_index]).cell.get_bravais_lattice().a
            workflow['tasks'][task] = total_energy_calculation(
                workflow['tasks'][task], workflow['run_options'])
        task_prev = task

    runtime.stop()
    return workflow


def total_energy_calculation(task, run_options):
    # Read atoms
    atoms = read(run_options['job_dir'] + '/' + run_options['atoms_dir'] + '/' +
                 task['atoms']['name'], format=run_options['software']['name'])
    # Initialize VASP calculation
    default_settings = vasplib.libdir(
        '/settings/' + run_options['software']['default_settings'])
    calc = vasplib.ini_vasp_calculation(
        default_settings, atoms, run_options['software']['settings'])
    # Set supercell volume and save atoms
    if type(atoms.cell.get_bravais_lattice()) == lattice.HEX:
        fac = 1 / np.sqrt(2)
    else:
        fac = 1

    calc.atoms.set_cell(
        calc.get_atoms().cell * task["lattice_param"] * fac, scale_atoms=True)
    initial_volume = calc.get_atoms().get_volume()
    task['atoms']['structure'] = calc.atoms.todict()

    # Execute Operations
    for op in task["operations"]:
        with open(vasplib.libdir('/operations/' + op + '.json'), 'r') as openfile:
            # Load task
            operation = json.load(openfile)
            # Attach additional information
            operation.update(task["operations"][op]['settings'])
            operation.update({"job_id": run_options["job_id"]})
            operation["outdir"] = run_options['result_dir'] + '/' + \
                task['atoms']['name'] + '/' + op
            # Execute task
            calc, value, unit = getattr(
                vasplib, operation["name"])(calc, operation)
            # Collect results
            task["operations"][op].update(
                {"result": {"value": value, "unit": unit}})
            # Write and plot results
            vasplib.result2json(operation, task, run_options, value, unit)
            if operation["makeplot"]:
                vasplib.makeplot(
                    operation['value'], value["etot"], operation['param'], operation['unit'])
                vasplib.saveplot(
                    operation["outdir"], run_options["job_id"] + "_" + operation['name'] + '_etot.png')

    return task
