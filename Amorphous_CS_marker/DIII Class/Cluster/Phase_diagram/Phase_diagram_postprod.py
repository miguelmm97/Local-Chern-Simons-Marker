# Collecting data and making plots for the marker and gap against width


import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from numpy.linalg import eigh
from numpy import pi
from random import sample
import argparse
import matplotlib.ticker as ticker
import os
import h5py
from functions import GaussianPointSet_3D, AmorphousHamiltonian3D_WT, local_marker_DIII, spectrum

Ms = np.linspace(-3.5, 3.5, 70)  # Mass parameter
Ws = np.linspace(0, 0.3, 70)
Rs = np.arange(4)  # Realisations
Ms_string = []
Ws_string = []


with open('PD_values_W.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
        params = line.split()
        Ws_string.append(params[0])


with open('PD_values_M.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
        params = line.split()
        Ms_string.append(params[0])


local_marker = np.zeros((len(Ws), len(Ms), len(Rs)))
gap = np.zeros((len(Ws), len(Ms), len(Rs)))

for file in os.listdir('outdir_pd'):
    if file.endswith('h5'):
        file_path = os.path.join("outdir_pd", file)
        with h5py.File(file_path, 'r') as f:

            datanode = f['data']
            value = datanode[()]
            M = datanode.attrs['M']
            width = datanode.attrs['width']
            R = datanode.attrs['R']
            string_M = '{:.4f}'.format(M)
            string_W = '{:.8f}'.format(width)

            m_index, w_index, sample = Ms_string.index(string_M), Ws_string.index(string_W), R
            local_marker[w_index, m_index, sample] = value[0]
            gap[w_index, m_index, sample] = value[1]






tol1 = 0.1
tol2 = 0.5
marker = np.mean(local_marker, axis=2)
gap = np.mean(gap, axis=2)
print(marker[0, :])

# Output data
file_name = "Phase_diagram_results.h5"
with h5py.File(file_name, 'w') as f:
    f.create_dataset("data", marker.shape, data=marker)

file_name = "Phase_diagram_gap.h5"
with h5py.File(file_name, 'w') as f:
    f.create_dataset("data", gap.shape, data=gap)


# gap_closing, gap_closing1, gap_closing2 = np.zeros((len(Ws), len(Ms))), np.zeros((len(Ws), len(Ms))), np.zeros((len(Ws), len(Ms)))
# gap_closing1[(tol1 < gap)] = 1
# gap_closing2[( gap < tol2)] = 1
# gap_closing = gap_closing1 * gap_closing2

# Clean part of the diagram
tol1 = 0.1
tol2 = 0.5
aux_matrix = np.zeros((len(Ws), len(Ms)))
aux_matrix[:, 0:-1] = gap[:, 1:]
dgap = np.zeros((len(Ws), len(Ms)))
gap_aux = aux_matrix - gap
dgap[gap_aux < 0] = 1
dgap[gap_aux > 0] = -1
aux_matrix2 = np.zeros((len(Ws), len(Ms)))
aux_matrix2[:, 0:-1] = dgap[:, 1:]
dgap2 = aux_matrix2 - dgap
gap_closing, gap_closing1, gap_closing2 = np.zeros((len(Ws), len(Ms))), np.zeros((len(Ws), len(Ms))), np.zeros((len(Ws), len(Ms)))
gap_closing1[(tol1 < gap)] = 1
gap_closing2[( gap < tol2)] = 1
gap_closing = gap_closing1 * gap_closing2


# # Amorphous part of the diagram
# tol3 = 0.01
# tol4 = 0.05
# aux_matrix1 = np.zeros((len(Ws), len(Ms)))
# aux_matrix1[:, 0:-1] = gap[:, 1:]
# dgap1 = np.zeros((len(Ws), len(Ms)))
# gap_aux1 = aux_matrix1 - gap
# dgap1[gap_aux1 < 0] = 1
# dgap1[gap_aux1 > 0] = -1
# gap_closing3, gap_closing4, gap_closing5 = np.zeros((len(Ws), len(Ms))), np.zeros((len(Ws), len(Ms))), np.zeros((len(Ws), len(Ms)))
# gap_closing4[(tol3 < gap)] = 1
# gap_closing5[( gap < tol4)] = 1
# gap_closing3 = gap_closing4 * gap_closing5


width_matrix = np.repeat(Ws, len(Ms))
mass_matrix = np.tile(Ms, len(Ws))

file_name = "Phase_diagram_mesh_x.h5"
with h5py.File(file_name, 'w') as f:
    f.create_dataset("data", width_matrix.shape, data=width_matrix)

file_name = "Phase_diagram_mesh_y.h5"
with h5py.File(file_name, 'w') as f:
    f.create_dataset("data", mass_matrix.shape, data=mass_matrix)


# Font format
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Colormap
colormap1 = plt.cm.coolwarm  # or any other colormap
normalize = mcolors.Normalize()

plt.figure(1)
plt.scatter(width_matrix, mass_matrix, c=marker, marker='s', cmap=colormap1,  linewidths=2.5)
contours = plt.contour(Ws, Ms, gap.T, levels=[0.2], colors="black")
plt.clabel(contours, inline=True, fontsize=12, fmt="$E_g \sim 0$")
#plt.colorbar(label=r'$\nu$')

# for i in range(len(Ms)):
#     for j in range(len(Ws)):
#         if Ws[j] < 0.09:
#             if gap_closing[j, i] == 1:
#                 if dgap[j, i] == 1:
#                     plt.scatter(Ws[j], Ms[i], c="Black", marker='.', linewidths=0.01)
#                 elif dgap[j, i] == -1:
#                     plt.scatter(Ws[j], Ms[i], c="Black", marker='.', linewidths=0.01)
#         else:
#             if gap_closing[j, i] == 1:
#                 if dgap1[j, i] == 1:
#                     plt.scatter(Ws[j], Ms[i], c="Black", marker='.', linewidths=0.01)
#                 elif dgap1[j, i] == -1:
#                     plt.scatter(Ws[j], Ms[i], c="Black", marker='.', linewidths=0.01)

#
# for i in range(len(Ms)):
#     for j in range(len(Ws)):
#         if gap_closing[j, i] == 1:
#              if np.abs(dgap2[j, i]) != 0:
#                     plt.scatter(Ws[j], Ms[i], c="Black", marker='.', linewidths=0.01)


# Axis labels and limits
plt.ylabel("$M$", fontsize=20)
plt.xlabel("$w$", fontsize=20)
plt.xlim(0, Ws[-1])
plt.ylim(Ms[0], Ms[-1])
# Axis tick


plt.tight_layout()
plt.savefig("try.eps", bbox_inches="tight")
plt.savefig("try.pdf", bbox_inches="tight")
plt.show()



#  Colormap
colormap2 = plt.cm.pink # or any other colormap
normalize = mcolors.Normalize(vmin=0, vmax=gap.max())


plt.figure(2)
plt.scatter(width_matrix, mass_matrix, c=gap, marker='s', cmap=colormap2,  norm=normalize, linewidths=2.5)
plt.colorbar(label=r'$\Delta_E$')


# Axis labels and limits
plt.ylabel("$M$", fontsize=20)
plt.xlabel("$w$", fontsize=20)
plt.xlim(0, Ws[-1])
plt.ylim(Ms[0], Ms[-1])
# Axis tick


plt.tight_layout()
plt.savefig("try2.eps", bbox_inches="tight")
plt.savefig("try2.pdf", bbox_inches="tight")
plt.show()