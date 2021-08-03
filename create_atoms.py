from ase.lattice.cubic import FaceCenteredCubic
from ase.calculators.vasp import Vasp
from ase.io import read
import vasplib

#General job information
job_info = {'subdir':'atoms',
            'atoms': 'Curtin',
            'atomtype': 'vasp'}

print('---------------------------')
print('Creating atoms: ' + job_info['atoms'] + ' ...')
print('---------------------------')

# #Create a cell and write a cell...
# atoms = FaceCenteredCubic(directions=[[1, 0, 0],
#                                         [0, 1, 0],
#                                         [0, 0, 1]],
#                         size=(1, 1, 1),
#                         symbol='Cu',
#                         latticeconstant=3.866)
# atoms[0].symbol = 'Pd'
# atoms[1].symbol = 'Ir'
# atoms[2].symbol = 'Pt'
# atoms.write(job_info['subdir']+'/'+job_info['atoms'],job_info['atomtype'])

# ... or load one
atoms = read(job_info['subdir']+'/'+job_info['atoms'],format=job_info['atomtype'])

#Get number of bands and cores
n_bands = vasplib.get_number_of_bands(atoms,job_info,f=1.0)
n_cores = vasplib.get_number_of_cores(n_bands,fac_bpc=8)


