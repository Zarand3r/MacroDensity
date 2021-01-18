#! /usr/bin/env python
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import macrodensity as md
# import argparse

# parser = argparse.ArgumentParser()
# # Input argument
# parser.add_argument('--input_file', default=f'{homedir}/task1/models/two_stream/test/test1.mp4', help='Input file to predict')
# parser.add_argument('--checkpoint_path', default=f'{settings1.checkpoints}/two_stream/AC_CE_EF_FG/_ckpt_epoch_34.ckpt')
# #parser.add_argument('--checkpoint_path', default=f'{settings1.checkpoints}/two_stream/AC_CE_EF_FG/_ckpt_epoch_46.ckpt', help='path to load checkpoints')
# parser.add_argument('--hparams_path', default=f'{homedir}/task1/models/two_stream/lightning_logs/AC_CE_EF_FG/alexnet_False_convLSTM/version_1/hparams.yaml', help='path to load hyperparameters')
# args = parser.parse_args()

job_identifier = "minimization1"
input_file = f'../../thesis/Slab/Doped/HER_H20/hydrogen_binding/0001/{job_identifier}/LOCPOT'
lattice_vector = 29.966237
output_file = f'../results/{job_identifier}_planar.dat'
# No need to alter anything after here
#------------------------------------------------------------------
# Get the potential
# This section should not be altered
#------------------------------------------------------------------
vasp_pot, NGX, NGY, NGZ, Lattice = md.read_vasp_density(input_file)
vector_a,vector_b,vector_c,av,bv,cv = md.matrix_2_abc(Lattice)
resolution_x = vector_a/NGX
resolution_y = vector_b/NGY
resolution_z = vector_c/NGZ
grid_pot, electrons = md.density_2_grid(vasp_pot,NGX,NGY,NGZ)
#------------------------------------------------------------------
## POTENTIAL
planar = md.planar_average(grid_pot,NGX,NGY,NGZ)
## MACROSCOPIC AVERAGE
macro  = md.macroscopic_average(planar,lattice_vector,resolution_z)
plt.plot(planar)
plt.plot(macro)
plt.savefig('Planar.eps')
plt.show()
plt.savefig(f"../results/{job_identifier}.jpg")
np.savetxt(output_file,planar)
##------------------------------------------------------------------
