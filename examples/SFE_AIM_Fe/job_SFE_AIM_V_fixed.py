# %%
import sys
import os
import json
import os
import numpy as np
from ase.calculators.vasp import Vasp
from ase.io import read
import numpy as np
import vasplib


def workflow_etot(settings):
    vasplib.prnt_header(
        'Running E_tot workflow: ' + settings['workflow_name'] + ' on cell ' + settings['atoms'])
    atoms = read(settings['job_dir'] + '/' + settings['atoms_dir'] + '/' +
                 settings['atoms'], format=settings['software'])

    # Initialize VASP calculation
    vasp_settings = vasplib.libdir('/settings/' + settings["software_settings"])
    calc = vasplib.ini_vasp_calculation(vasp_settings,atoms, settings["vasp_settings"])

    # Set volume
    calc.atoms.set_cell(
        calc.get_atoms().cell * settings["lattice_param"], scale_atoms=True)
    initial_volume = calc.get_atoms().get_volume()

    # Execute tasks
    for t in settings["tasks"]:
        with open(vasplib.libdir('/tasks/' + t + '.json'), 'r') as openfile:
            # Load task
            task = json.load(openfile)
            # Attach additional information
            task.update(settings['task_input'])
            task["outdir"] = settings['job_dir'] + '/' + settings['result_dir'] + '/' + \
                settings['atoms'] + '/' + t

            # Execute task
            calc, result, unit = getattr(vasplib, task["name"])(calc, task)
            # Write and plot results
            vasplib.result2json(task, settings, result, unit)
            if task["makeplot"]:
                vasplib.makeplot(
                    task['value'], result["etot"], task['param'], task['unit'])
                vasplib.saveplot(
                    task["outdir"], settings["job_id"] + "_" + task['name'] + '_etot.png')

    return result, unit


def workflow_post(settings):
    vasplib.prnt_header(
        'Running Post workflow: ' + settings['workflow_name'])

    # Execute tasks
    for t in settings["tasks"]:
        with open(vasplib.libdir('/tasks/' + t + '.json'), 'r') as openfile:
            # Load task
            task = json.load(openfile)
            # Attach additional information
            task.update(settings['task_input'])
            task["outdir"] = settings['job_dir'] + '/' + settings['result_dir'] + '/' + \
                settings['workflow_name'] + '/' + t
            # Execute task
            result, unit = getattr(vasplib, task["name"])(task)
            # Write and plot results
            vasplib.result2json(task, settings, result, unit)
            if task["makeplot"]:
                vasplib.makeplot(
                    task['value'], result["etot"], task['param'], task['unit'])
                vasplib.saveplot(
                    task["outdir"], settings["job_id"] + "_" + task['name'] + '_etot.png')

    return result, unit


if __name__ == '__main__':

    # Global settings
    settings = {'atoms_dir': 'atoms',
                'job_dir': os.path.dirname(__file__), 
                'software_settings': 'vasp_default',
                'result_dir': 'results/'+os.path.basename(sys.argv[0]).split('.')[0],
                'software': 'vasp'
                }
    settings["job_id"] = vasplib.make_jobid(0)
    settings["ncore"], settings["kpar"] = vasplib.get_par_settings(sys.argv)

    # Log command line output
    sys.stdout = vasplib.Logger(
        settings['job_dir'] + '/' + settings["result_dir"] + "/" + settings["job_id"] + "_log.output")
    print(' /////  AIM fcc - hcp calculations - ' + settings["job_id"])

    # FCC workflow settings
    settings_fcc = {
        'workflow_name': 'Relax fcc',
        'atoms': 'fcc',
        'lattice_param': 3.573,
        'task_input': {'job_id': settings["job_id"]},
        'tasks': ['kpoint_optimization']}
    settings_fcc.update(settings)

    # FCC workflow
    result, unit = workflow_etot(settings_fcc)

    # HCP workflow settings
    settings_hcp = {
        'workflow_name': 'Relax hcp',
        'atoms': 'hcp',
        'lattice_param': (result["v0"][-1] * 4)**(1 / 3) / np.sqrt(2),
        'task_input': {'job_id': settings["job_id"]},
        'tasks': ['kpoint_optimization', 'cell_relaxation_isif2_coa', 'fit_eos_isif2_coa']}
    settings_hcp.update(settings)

    # HCP workflow
    result, unit = workflow_etot(settings_hcp)

    # SFE AIM workflow settings
    settings_sfe_aim = {
        'workflow_name': 'SFE AIM',
        'tasks': ['sfe_aim'],
        'task_input': {
            'software': settings['software'],
            'atoms_dir': settings['job_dir'] + "/" + settings['atoms_dir'],
            'result_dir': settings['job_dir'] + "/" + settings["result_dir"],
            'job_id': settings["job_id"],
            # parent, child - order matters!
            'atoms': ['fcc', 'hcp'],
            'etot_source': ['kpoint_optimization', 'fit_eos_isif2_coa'],
            'etot_job': [settings["job_id"], settings["job_id"]],
            'etot_index': [-1, -1]
        }
    }
    settings_sfe_aim.update(settings)

    # SFE AIM workflow
    result, unit = workflow_post(settings_sfe_aim)

# %%
