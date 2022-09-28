import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from colorbar_functions import hex_to_rgb, rgb_to_dec,get_continuous_cmap #import functions for colormap
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import matplotlib.ticker as ticker
import os
import h5py

# Import data
outdir="."
for file in os.listdir(outdir):
    if file == "Sup_material_avmarker.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            av_marker = datanode[()]

    if file == "Sup_material_marker.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            marker = datanode[()]

    if file == "Sup_material_xaxis.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            x_axis = datanode[()]

    if file == "Sup_material_yaxis.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            y_axis = datanode[()]



font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 13, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=20)
axcolour = ['#FF416D', '#3F6CFF', '#00B5A1'] # light


divnorm = mcolors.TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=max(marker))
hex_list = ['#ff416d', '#ff7192', '#ffa0b6', '#ffd0db', '#ffffff', '#cfdaff', '#9fb6ff', '#6f91ff', '#3f6cff']

# Phase diagram
fig, ax = plt.subplots(figsize=(8, 6))
scatters = ax.scatter(x_axis, y_axis, c=marker, marker='.', norm=divnorm, cmap = get_continuous_cmap(hex_list),  linewidths=2.5)
cbar = plt.colorbar(scatters, ax=ax)

ax.set_ylabel("$y$", fontsize=25)
ax.set_xlabel("$x$", fontsize=25)
cbar.set_label(label='$\\nu$', size=25, labelpad=-15, y=0.5)
ax.set_xlim(-0.5, 12 - 0.5)
ax.set_ylim(-0.5, 12 - 0.5)

plt.savefig("Sup_material.pdf", bbox_inches="tight")
plt.show()

