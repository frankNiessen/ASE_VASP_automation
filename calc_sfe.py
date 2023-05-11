# %% Get data
import numpy as np
import re
import os.path

# %% 2023_04_25_Fe_SFE_9Layer_V0
outfile_fcc = "2023_04_25_Fe_SFE_9Layer_V0/output/Output-6216662.out"
outfile_SF = "2023_04_25_Fe_SFE_9Layer_V0/output/Output-6216811.out"
nr_atoms = 9
rel_area = 1

# %% 2023_04_25_Fe_SFE_6Layer_V0
outfile_fcc = "2023_04_25_Fe_SFE_6Layer_V0/output/Output-6242730.out"
outfile_SF = "2023_04_25_Fe_SFE_6Layer_V0/output/Output-6242731.out"
nr_atoms = 6
rel_area = 1

# %% 2023_05_01_Fe_SFE_6Layer_54at_Abbasi
outfile_fcc = "2023_05_01_Fe_SFE_6Layer_54at_Abbasi/output/Output-6243380.out"
outfile_SF = "2023_05_01_Fe_SFE_6Layer_54at_Abbasi/output/Output-6243381.out"
nr_atoms = 54
rel_area = 3*3
# %% Get data
v0 = []
e0 = []
B = []
for file in [outfile_fcc,outfile_SF]:
    with open(file) as f:
        for line in f:
            pass
        last_line = line
        num_data = [float(s) for s in re.findall(r'-?[0-9]+\.[0-9]+', last_line )]
        v0.append(num_data[0])
        e0.append(num_data[1])
# %% Calc and save SFE value
a = (v0[0]/nr_atoms*4)**(1/3) #Lattice constant [AA]
area = rel_area*np.sqrt(3)/2*(np.sqrt(2)/2*a)**2
sfe = (e0[1]-e0[0]) * 1.602177e-16 / (area * 1e-20)
sfe_str = "SFE:{:10.2f} mJ.m^(-2)".format(sfe)
print(sfe_str)
with open(os.path.dirname(outfile_SF) + "/SFE.txt", "w") as text_file:
    text_file.write(sfe_str)

# %%
