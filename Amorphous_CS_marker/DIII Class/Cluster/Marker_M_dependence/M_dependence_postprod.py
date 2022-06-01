# Collecting data and making plots for the marker against M


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
Ms = np.linspace(-4, 4, 50)  # Mass parameter
Rs = np.arange(100)  # Realisations
Ms_inset = [0, 2, 4]


local_marker_0 = np.zeros((len(Ms), len(Rs)))
local_marker_1 = np.zeros((len(Ms), len(Rs)))
local_marker_2 = np.zeros((len(Ms), len(Rs)))
site_marker_0 = np.zeros((12,))
site_marker_1 = np.zeros((12,))
site_marker_2 = np.zeros((12,))

for file in os.listdir('.'):
    if file.endswith('h5'):
        with h5py.File(file, 'r') as f:

            datanode = f['data']
            value = datanode[()]
            size = datanode.attrs['size']
            M = datanode.attrs['M']
            R = datanode.attrs['R']
            width = datanode.attrs['width']

            if size == Ls[0]:
                row, col = np.where(Ms == M), R
                local_marker_0[row, col] = value

            if size == Ls[1]:
                row, col = np.where(Ms == M), R
                local_marker_1[row, col] = value

            if size == Ls[2]:
                row, col = np.where(Ms == M), R
                local_marker_2[row, col] = value


            if size == "inset":
                if M == Ms_inset[0]:
                    site_marker_0 = site_marker_0 + value
                elif M == Ms_inset[1]:
                    site_marker_1 = site_marker_1 + value
                elif M == Ms_inset[2]:
                    site_marker_2 = site_marker_2 + value



marker_0 = np.mean(local_marker_0, axis=1)
marker_1 = np.mean(local_marker_1, axis=1)
marker_2 = np.mean(local_marker_2, axis=1)
std_0 = np.std(local_marker_0, axis=1)
std_1 = np.std(local_marker_1, axis=1)
std_2 = np.std(local_marker_2, axis=1)
site_marker_0 = site_marker_0 / len(Rs)
site_marker_1 = site_marker_1 / len(Rs)
site_marker_2 = site_marker_2 / len(Rs)

# %% Plots

# Font format
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Main figure
fig, ax = plt.subplots(figsize=(8, 6))
axcolour = ['#365365', '#71250E', '#6F6534']
ax.plot(Ms, marker_0, color=axcolour[0], marker='.', markersize=12, label='$L= $'+ str(Ls[0]))
ax.plot(Ms, marker_1, color=axcolour[1], marker='.', markersize=12, label='$L= $'+ str(Ls[1]))
ax.plot(Ms, marker_2, color=axcolour[2], marker='.', markersize=12, label='$L= $'+ str(Ls[2]))

# Axis labels and limits
ax.set_ylabel("$cs$", fontsize=20)
ax.set_xlabel("$M/t_1$", fontsize=20)
ax.set_xlim(-4, 4)
ax.set_ylim(-2.2, 1.2)

# Axis ticks
ax.tick_params(which='major', width=0.75)
ax.tick_params(which='major', length=14)
ax.tick_params(which='minor', width=0.75)
ax.tick_params(which='minor', length=7)
majorsy = [-2, -1, 0, 1]
minorsy = [-2.2, -1.5, -0.5, 0.5, 1.2]
majorsx = [-4,  -2,  0,  2, 4]
minorsx = [-3, -1, 1, 3]
ax.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
ax.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
ax.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
ax.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))

# Legend and inset text
ax.legend(loc='upper right', frameon=False)
ax.text(-3.5, -1.75, " $w=$" + str(width), fontsize=20)
ax.text(-3.5, -2, " $N_{\\rm{samples}}=$" + str(len(Rs)), fontsize=20)

plt.tight_layout()
# plt.title(" $N_{\\rm{samples}}=$" + str(n_realisations) + " $w=$" + str(width),fontsize=20)



# Inset figure
# Placement and inset
left, bottom, width, height = [0.76, 0.3, 0.2, 0.2]
inset_ax = fig.add_axes([left, bottom, width, height])
insetcolour = ['#BF7F04', '#BF5B05', '#8C1C04']

inset_ax.plot(range(0, len(site_marker_0)), site_marker_0, color=insetcolour[0], marker='.',
                                                                     markersize=8, label='$M=$' + str(Ms_inset[0]))
inset_ax.plot(range(0, len(site_marker_1)), site_marker_1, color=insetcolour[1], marker='.',
                                                                     markersize=8, label='$M=$' + str(Ms_inset[1]))
inset_ax.plot(range(0, len(site_marker_2)), site_marker_2, color=insetcolour[2], marker='.',
                                                                     markersize=8, label='$M=$' + str(Ms_inset[2]))

# Axis labels and limits
inset_ax.set_ylabel("$cs$", fontsize=20)
inset_ax.set_xlabel("$x$", fontsize=20)
inset_ax.set_xlim(0, 11)
inset_ax.set_ylim(-2.5, 2)

# Axis ticks
inset_ax.tick_params(which='major', width=0.75)
inset_ax.tick_params(which='major', length=14)
inset_ax.tick_params(which='minor', width=0.75)
inset_ax.tick_params(which='minor', length=7)
majorsy2 = [-2, -1, 0, 1, 2]
# minorsy2 = [-2.2, -1.5, -0.5, 0.5, 1.2]
majorsx2 = [0, 5.5, 11]
minorsx2 = [2.75, 8.25]
inset_ax.yaxis.set_major_locator(ticker.FixedLocator(majorsy2))
# inset_ax.yaxis.set_minor_locator(ticker.FixedLocator(minorsy2))
inset_ax.xaxis.set_major_locator(ticker.FixedLocator(majorsx2))
inset_ax.xaxis.set_minor_locator(ticker.FixedLocator(minorsx2))

# Legend
inset_ax.legend(loc=(0, 0.9), mode="expand", ncol=3, frameon=False)

plt.show()

