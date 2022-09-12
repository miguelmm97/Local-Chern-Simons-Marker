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

    if file == "Qgap_t2.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            Qgap = datanode[:]

    if file == "Egap_t2.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            Egap = datanode[:]

    if file == "gap_t2_xaxis.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            inset_xaxis = datanode[:]
# %% Plots


# Font format
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 13, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=20)



# Main figure
fig, ax = plt.subplots(figsize=(8, 6))
axcolour = ['#FF416D', '#3F6CFF', '#00B5A1'] # light
ax.plot(Marker_width_xaxis, marker_width_8, color=axcolour[0], marker='.', markersize=12, label='$L= 8$')
ax.plot(Marker_width_xaxis, marker_width_10, color=axcolour[1], marker='.', markersize=12, label='$L= 10$')
ax.plot(Marker_width_xaxis, marker_width_12, color=axcolour[2], marker='.', markersize=12, label='$L= 12$')


# Axis labels and limits
ax.set_ylabel("$\\nu_{\\rm{cs}}$", fontsize=25)
ax.set_xlabel("$w$", fontsize=25)
ax.set_xlim(0, 0.2)
ax.set_ylim(-0.1, 1.1)
ax.yaxis.set_label_coords(-0.13, 0.5)

# Axis ticks
ax.tick_params(which='major', width=0.75)
ax.tick_params(which='major', length=14)
ax.tick_params(which='minor', width=0.75)
ax.tick_params(which='minor', length=7)
majorsy = [0, 0.5, 1]
minorsy = [0.25, 0.75, 1.25]
majorsx = [0, 0.05, 0.1, 0.15, 0.2]
minorsx = [0.025, 0.075, 0.125, 0.175]
ax.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
ax.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
ax.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
ax.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))

# Legend and inset text
ax.text(0.145, 0.85, "$t=$ $0.5$", fontsize=25)
ax.legend(loc=(0.67, 0.5), frameon=False, fontsize=20)


# Inset figure
# Placement and inset
# left, bottom, width, height = [0.72, 0.27, 0.24, 0.26]
left, bottom, width, height = [0.06, 0.15, 0.42, 0.42]
inset_ax = inset_axes(ax, height="100%", width="100%", bbox_to_anchor = [left, bottom, width, height], bbox_transform = ax.transAxes, loc = 3 ) # [left, bottom, width, height])
#insetcolour = ['#BF7F04', '#BF5B05', '#8C1C04'] # dark
insetcolour = ['#6668FF', '#FFC04D','#FF7D66'] # light

inset_ax.plot(inset_xaxis, Egap, color=insetcolour[2], linestyle="solid", linewidth=3, label="$\Delta$")
inset_ax.plot(inset_xaxis, Qgap, color=insetcolour[1], linestyle="solid", linewidth=3, label="$\Delta i[\\rho,R]$")

# Axis labels and limits
# inset_ax.set_ylabel("gap", fontsize=20)
inset_ax.set_xlabel("$t$", fontsize=20)
inset_ax.set_xlim(0, 2)
inset_ax.set_ylim(0, 2)

# Axis labels
inset_ax.xaxis.set_label_coords(0.5, -0.2)
inset_ax.yaxis.set_label_coords(-0.13, 0.5)
# Axis ticks
inset_ax.tick_params(which='major', width=0.75, labelsize=15)
inset_ax.tick_params(which='major', length=7, labelsize=15)
inset_ax.tick_params(which='minor', width=0.75)
inset_ax.tick_params(which='minor', length=3.5)
majorsy2 = [0, 1, 2]
minorsy2 = [0.5, 1.5]
majorsx2 = [0, 1, 2]
minorsx2 = [0.5, 1.5]
inset_ax.yaxis.set_major_locator(ticker.FixedLocator(majorsy2))
inset_ax.yaxis.set_minor_locator(ticker.FixedLocator(minorsy2))
inset_ax.xaxis.set_major_locator(ticker.FixedLocator(majorsx2))
inset_ax.xaxis.set_minor_locator(ticker.FixedLocator(minorsx2))

inset_ax.legend(loc="upper right", frameon=False, fontsize=15)

# plt.title(" $N_{\\rm{samples}}=$" + str(n_realisations) + " $w=$" + str(width),fontsize=20)
plt.tight_layout()
plt.savefig("try1.pdf", bbox_inches="tight")
plt.show()

