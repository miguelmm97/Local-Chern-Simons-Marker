import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import matplotlib.ticker as ticker
import os
import h5py

# Import data
outdir="."
for file in os.listdir(outdir):
    if file == "Marker_width_8_AII_results.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            marker_width_8 = datanode[:]
            
    if file == "Marker_width_10_AII_results.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            marker_width_10 = datanode[:]
            
    if file == "Marker_width_12_AII_results.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            marker_width_12 = datanode[:]

    if file == "Marker_width_Xaxis_AII_results.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            Marker_width_xaxis = datanode[:]


# %% Plots


# Figure 1
# Font format
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Main figure
fig, ax = plt.subplots(figsize=(8, 6))
axcolour = ['#365365', '#6F6534', '#9932CC']
ax.plot(Marker_width_xaxis, marker_width_8, color=axcolour[0], marker='.', markersize=12, label='$L= 8$' + " $t_2=0.5$")
ax.plot(Marker_width_xaxis, marker_width_10, color=axcolour[1], marker='.', markersize=12, label='$L= 10$' + " $t_2=0.5$")
ax.plot(Marker_width_xaxis, marker_width_12, color=axcolour[2], marker='.', markersize=12, label='$L= 12$' + " $t_2=0.5$")
ax.plot(Marker_width_xaxis, np.repeat(0, len(Marker_width_xaxis)), '--')
ax.plot(Marker_width_xaxis, np.repeat(1, len(Marker_width_xaxis)), '--')

# Axis labels and limits
ax.set_ylabel("$\\nu$", fontsize=20)
ax.set_xlabel("$w$", fontsize=20)
ax.set_xlim(0, 0.2)
ax.set_ylim(-0.1, 1.4)

# Axis ticks
#ax.tick_params(which='major', width=0.75)
#ax.tick_params(which='major', length=14)
ax.tick_params(which='minor', width=0.75)
ax.tick_params(which='minor', length=7)
#majorsy = [0, 1, 2]
#minorsy = [-0.2, 0.5, 1, 1.5, 2.2]
majorsx = [0, 0.05, 0.1, 0.15, 0.2]
minorsx = [0.025, 0.075, 0.125, 0.175]
#ax.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
#ax.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
ax.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
ax.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))

# Legend and inset text
ax.legend(loc='upper right', frameon=False, fontsize=20)
# ax.text(1.5, 0.5, " $M=$" + str(M), fontsize=20)
# ax.text(1.5, 0.4," $S= -\sigma_0 \otimes \sigma_y$", fontsize=20)
# ax.text(-4.5, -1.5, " $N=$" + str(n_realisations), fontsize=20)

# plt.title(" $N_{\\rm{samples}}=$" + str(n_realisations) + " $w=$" + str(width),fontsize=20)
plt.savefig("try1.pdf", bbox_inches="tight")
plt.show()

