# Collecting data and making plots for the marker and gap against width


import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from numpy import pi
from random import sample
import argparse
import matplotlib.ticker as ticker
import os
import h5py
from functions import GaussianPointSet_3D, AmorphousHamiltonian3D_WT, local_marker_DIII, spectrum

Ls = [8, 10, 12]  # System sizes
Rs = np.arange(4)  # Realisations
width1 = np.linspace(0, 0.2, 50)
width2 = np.linspace(0.2, 0.3, 25)
Ws = np.append(width1[:-1], width2)
Ws_string = []
Ms = [0, 2, 4]


with open('width_values.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
        params = line.split()
        Ws_string.append(params[0])

local_marker_0 = np.zeros((len(Ms), len(Ws), len(Rs)))
local_marker_1 = np.zeros((len(Ms), len(Ws), len(Rs)))
local_marker_2 = np.zeros((len(Ms), len(Ws), len(Rs)))
gap_0 = np.zeros((len(Ms), len(Ws), len(Rs)))
gap_1 = np.zeros((len(Ms), len(Ws), len(Rs)))
gap_2 = np.zeros((len(Ms), len(Ws), len(Rs)))
cont = 0
for file in os.listdir('outdir_width'):

    if file.endswith('h5'):

        file_path = os.path.join("outdir_width", file)

        with h5py.File(file_path, 'r') as f:

            datanode = f['data']
            value = datanode[()]
            size = datanode.attrs['size']
            M = datanode.attrs['M']
            R = datanode.attrs['R']
            width = datanode.attrs['width']
            width_string = '{:.8f}'.format(width)


            if size == Ls[0]:
                m_index, row, col = np.where(Ms == M), Ws_string.index(width_string), R
                local_marker_0[m_index, row, col] = value[0]
                gap_0[m_index, row, col] = value[1]

            if size == Ls[1]:
                m_index, row, col = np.where(Ms == M), Ws_string.index(width_string), R
                local_marker_1[m_index, row, col] = value[0]
                gap_1[m_index, row, col] = value[1]


            if size == Ls[2]:
                print(R, M, width)

                if R>3:
                    R=3

                m_index, row, col = np.where(Ms == M), Ws_string.index(width_string), R
                local_marker_2[m_index, row, col] = value[0]
                gap_2[m_index, row, col] = value[1]


marker_0 = np.mean(local_marker_0, axis=2)
marker_1 = np.mean(local_marker_1, axis=2)
marker_2 = np.mean(local_marker_2, axis=2)
std_0 = np.std(local_marker_0, axis=2)
std_1 = np.std(local_marker_1, axis=2)
std_2 = np.std(local_marker_2, axis=2)
gap_mean_0 = np.mean(gap_0, axis=2)
gap_mean_1 = np.mean(gap_1, axis=2)
gap_mean_2 = np.mean(gap_2, axis=2)
std_gap_0 = np.std(gap_0, axis=2)
std_gap_1 = np.std(gap_1, axis=2)
std_gap_2 = np.std(gap_2, axis=2)


# %% Plots

# Font format
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Main figure
fig, ax = plt.subplots(figsize=(8, 6))
axcolour = ['#C00425', '#306367']
axmarkers = ['dashed', 'dotted', 'solid']
ax.plot(Ws, marker_0[0, :], color='k', linestyle=axmarkers[0], linewidth=2, label='$L= $' + str(Ls[0]))
ax.plot(Ws, marker_0[0, :], color=axcolour[0], linestyle=axmarkers[0], linewidth=2)
ax.plot(Ws, marker_1[0, :], color='k', linestyle=axmarkers[1], linewidth=2, label='$L= $' + str(Ls[1]))
ax.plot(Ws, marker_1[0, :], color=axcolour[0], linestyle=axmarkers[1], linewidth=2)
ax.plot(Ws, marker_2[0, :], color='k', linestyle=axmarkers[2], linewidth=2, label='$L= $' + str(Ls[2]))
ax.plot(Ws, marker_2[0, :], color=axcolour[0], linestyle=axmarkers[2], linewidth=2)


# Axis labels and limits
ax.set_ylabel("$cs$", fontsize=20)
ax.set_xlabel("$w$", fontsize=20)
ax.set_xlim(0, 0.3)
ax.set_ylim(-2.2, 0.2)

# Axis ticks
ax.tick_params(which='major', width=0.75)
ax.tick_params(which='major', length=14)
ax.tick_params(which='minor', width=0.75)
ax.tick_params(which='minor', length=7)
majorsy = [-2, -1.5, -1, -0.5, 0]
minorsy = [-1.75, -1.25, -0.75, -0.25]
majorsx = [0, 0.1, 0.2]
minorsx = [0.05, 0.15]
ax.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
ax.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
ax.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
ax.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))

ax.arrow(0.025, -1.5, -0.02, 0, length_includes_head=True,
         head_width=0.05, head_length=0.005, color='#C00425')

ax.arrow(0.175, -1.75, 0.02, 0, length_includes_head=True,
         head_width=0.05, head_length=0.005, color='#306367')

# Legend and inset text
ax.legend(loc='upper center', frameon=False, fontsize=20)

right_ax = ax.twinx()
right_ax.plot(Ws, gap_mean_0[0, :], color=axcolour[1], linestyle=axmarkers[0], linewidth=2)
right_ax.plot(Ws, gap_mean_1[0, :], color=axcolour[1], linestyle=axmarkers[1], linewidth=2)
right_ax.plot(Ws, gap_mean_2[0, :], color=axcolour[1], linestyle=axmarkers[2], linewidth=2)
right_ax.set_ylabel("$E_g/t_1$", fontsize=20)
right_ax.set_ylim(-0.1, 2)

# Axis ticks
right_ax.tick_params(which='major', width=0.75)
right_ax.tick_params(which='major', length=14)
right_ax.tick_params(which='minor', width=0.75)
right_ax.tick_params(which='minor', length=7)
majorsy = [0, 0.5, 1, 1.5, 2]
minorsy = [0.25, 0.75, 1.25, 1.75]
majorsx = [0, 0.1, 0.2]
minorsx = [0.05, 0.15]
right_ax.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
right_ax.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
right_ax.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
right_ax.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))

plt.tight_layout()
plt.savefig("try1.eps", bbox_inches="tight")
plt.savefig("try1.pdf", bbox_inches="tight")
# plt.title(" $N_{\\rm{samples}}=$" + str(n_realisations) + " $w=$" + str(width),fontsize=20)
plt.show()







# Main figure
fig2, ax2 = plt.subplots(figsize=(8, 6))
axcolour = ['#C00425', '#306367']
axmarkers = ['dashed', 'dotted', 'solid']
ax2.plot(Ws, marker_0[1, :], color='k', linestyle=axmarkers[0], linewidth=2, label='$L= $' + str(Ls[0]))
ax2.plot(Ws, marker_0[1, :], color=axcolour[0], linestyle=axmarkers[0], linewidth=2)
ax2.plot(Ws, marker_1[1, :], color='k', linestyle=axmarkers[1], linewidth=2, label='$L= $' + str(Ls[1]))
ax2.plot(Ws, marker_1[1, :], color=axcolour[0], linestyle=axmarkers[1], linewidth=2)
ax2.plot(Ws, marker_2[1, :], color='k', linestyle=axmarkers[2], linewidth=2, label='$L= $' + str(Ls[2]))
ax2.plot(Ws, marker_2[1, :], color=axcolour[0], linestyle=axmarkers[2], linewidth=2)

# Axis labels and limits
ax2.set_ylabel("$cs$", fontsize=20)
ax2.set_xlabel("$w$", fontsize=20)
ax2.set_xlim(0, 0.3)
ax.set_ylim(-0.2, 2.2)

# Axis ticks
ax2.tick_params(which='major', width=0.75)
ax2.tick_params(which='major', length=14)
ax2.tick_params(which='minor', width=0.75)
ax2.tick_params(which='minor', length=7)
majorsy = [0, 0.5, 1, 1.5, 2]
minorsy = [0.25, 0.75, 1.25, 1.75]
majorsx = [0, 0.1, 0.2]
minorsx = [0.05, 0.15]
ax2.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
ax2.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
ax2.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
ax2.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))

ax2.arrow(0.025, 0.75, -0.02, 0, length_includes_head=True,
         head_width=0.05, head_length=0.005, color='#C00425')

ax2.arrow(0.175, 0.6, 0.02, 0, length_includes_head=True,
         head_width=0.05, head_length=0.005, color='#306367')

# Legend and inset text
ax2.legend(loc='upper right', frameon=False, fontsize=20)

right_ax = ax2.twinx()
right_ax.plot(Ws, gap_mean_0[1, :], color=axcolour[1], linestyle=axmarkers[0], linewidth=2)
right_ax.plot(Ws, gap_mean_1[1, :], color=axcolour[1], linestyle=axmarkers[1], linewidth=2)
right_ax.plot(Ws, gap_mean_2[1, :], color=axcolour[1], linestyle=axmarkers[2], linewidth=2)
right_ax.set_ylabel("$E_g/t_1$", fontsize=20)
right_ax.set_ylim(-0.1, 2)

# Axis ticks
right_ax.tick_params(which='major', width=0.75)
right_ax.tick_params(which='major', length=14)
right_ax.tick_params(which='minor', width=0.75)
right_ax.tick_params(which='minor', length=7)
majorsy = [0, 0.5, 1, 1.5, 2]
minorsy = [0.25, 0.75, 1.25, 1.75]
majorsx = [0, 0.1, 0.2]
minorsx = [0.05, 0.15]
right_ax.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
right_ax.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
right_ax.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
right_ax.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))

plt.tight_layout()
plt.savefig("try2.eps", bbox_inches="tight")
plt.savefig("try2.pdf", bbox_inches="tight")
# plt.title(" $N_{\\rm{samples}}=$" + str(n_realisations) + " $w=$" + str(width),fontsize=20)
plt.show()







# Main figure
fig3, ax3 = plt.subplots(figsize=(8, 6))
axcolour = ['#C00425', '#306367']
axmarkers = ['dashed', 'dotted', 'solid']
ax3.plot(Ws, marker_0[2, :], color='k', linestyle=axmarkers[0], linewidth=2, label='$L= $' + str(Ls[0]))
ax3.plot(Ws, marker_0[2, :], color=axcolour[0], linestyle=axmarkers[0], linewidth=2)
ax3.plot(Ws, marker_1[2, :], color='k', linestyle=axmarkers[1], linewidth=2, label='$L= $' + str(Ls[1]))
ax3.plot(Ws, marker_1[2, :], color=axcolour[0], linestyle=axmarkers[1], linewidth=2)
ax3.plot(Ws, marker_2[2, :], color='k', linestyle=axmarkers[2], linewidth=2, label='$L= $' + str(Ls[2]))
ax3.plot(Ws, marker_2[2, :], color=axcolour[0], linestyle=axmarkers[2], linewidth=2)

# Axis labels and limits
ax3.set_ylabel("$cs$", fontsize=20)
ax3.set_xlabel("$w$", fontsize=20)
ax3.set_xlim(0, 0.3)
ax.set_ylim(-1, 1)

# Axis ticks
ax3.tick_params(which='major', width=0.75)
ax3.tick_params(which='major', length=14)
ax3.tick_params(which='minor', width=0.75)
ax3.tick_params(which='minor', length=7)
majorsy = [-1, -0.5, 0, 0.5, 1]
minorsy = [-0.75, -0.25, 0.25, 0.75]
majorsx = [0, 0.1, 0.2]
minorsx = [0.05, 0.15]
ax3.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
ax3.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
ax3.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
ax3.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))

ax3.arrow(0.025, 0.25, -0.02, 0, length_includes_head=True,
         head_width=0.05, head_length=0.005, color='#C00425')

ax3.arrow(0.175, 0.5, 0.02, 0, length_includes_head=True,
         head_width=0.05, head_length=0.005, color='#306367')

# Legend and inset text
ax3.legend(loc='upper right', frameon=False, fontsize=20)

right_ax = ax3.twinx()
right_ax.plot(Ws, gap_mean_0[2, :], color=axcolour[1], linestyle=axmarkers[0], linewidth=2)
right_ax.plot(Ws, gap_mean_1[2, :], color=axcolour[1], linestyle=axmarkers[1], linewidth=2)
right_ax.plot(Ws, gap_mean_2[2, :], color=axcolour[1], linestyle=axmarkers[2], linewidth=2)
right_ax.set_ylabel("$E_g/t_1$", fontsize=20)
right_ax.set_ylim(2, 4)

# Axis ticks
right_ax.tick_params(which='major', width=0.75)
right_ax.tick_params(which='major', length=14)
right_ax.tick_params(which='minor', width=0.75)
right_ax.tick_params(which='minor', length=7)
majorsy = [2, 2.5, 3, 3.5, 4]
minorsy = [2.25, 2.75, 3.25, 3.75]
majorsx = [0, 0.1, 0.2]
minorsx = [0.05, 0.15]
right_ax.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
right_ax.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
right_ax.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
right_ax.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))

plt.tight_layout()
plt.savefig("try3.eps", bbox_inches="tight")
plt.savefig("try3.pdf", bbox_inches="tight")
# plt.title(" $N_{\\rm{samples}}=$" + str(n_realisations) + " $w=$" + str(width),fontsize=20)
plt.show()
