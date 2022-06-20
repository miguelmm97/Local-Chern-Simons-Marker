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
    if file == "Marker_M_results.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            marker_M_8 = datanode[0, :]
            marker_M_10 = datanode[1, :]
            marker_M_12 = datanode[2, :]

    if file == "Phase_diagram_results.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            phase_diagram= datanode[()]

    if file == "Phase_diagram_gap.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            pd_gap= datanode[()]

    if file == "Marker_width_8_results.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            marker_width_8_M0 = datanode[0, :]
            marker_width_8_M2 = datanode[1, :]

    if file == "Marker_width_10_results.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            marker_width_10_M0 = datanode[0, :]
            marker_width_10_M2 = datanode[1, :]

    if file == "Marker_width_12_results.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            marker_width_12_M0 = datanode[0, :]
            marker_width_12_M2 = datanode[1, :]

    if file == "Marker_width_8_gap_results.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            gap_width_8_M0 = datanode[0, :]
            gap_width_8_M2 = datanode[1, :]

    if file == "Marker_width_10_gap_results.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            gap_width_10_M0 = datanode[0, :]
            gap_width_10_M2 = datanode[1, :]

    if file == "Marker_width_12_gap_results.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            gap_width_12_M0 = datanode[0, :]
            gap_width_12_M2 = datanode[1, :]

    if file == "Marker_M_Xaxis_results.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            Marker_M_xaxis = datanode[:]

    if file == "Marker_width_Xaxis_results.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            Marker_width_xaxis = datanode[:]

    if file == "Phase_diagram_mesh_x.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            pd_mesh_x = datanode[()]

    if file == "Phase_diagram_mesh_y.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            pd_mesh_y = datanode[()]

    if file == "Phase_diagram_gap_x.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            pd_gap_x = datanode[()]

    if file == "Phase_diagram_gap_y.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            pd_gap_y = datanode[()]

    if file == "Marker_M_inset_M0.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            M_inset_0 = datanode[()]

    if file == "Marker_M_inset_M2.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            M_inset_2 = datanode[()]

    if file == "Marker_M_inset_M4.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            M_inset_4 = datanode[()]


M_inset_xaxis =np.arange(1, 13)


#%% FIgure grid

# Font format
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 28, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=25)
fig, (ax2, ax1, ax3) = plt.subplots(1, 3, figsize=(27, 8))
fig.subplots_adjust(left=0.04, bottom=0.15, right=1, top=0.95, wspace=0.3, hspace=0.4)


#%% Marker vs Width


# Marker
# fig1, ax1 = plt.subplots(figsize=(8, 6))
axcolour = ['#FF416D','#6668FF']
axmarkers = ['dashed', 'dotted', 'solid']
# ax.plot(Marker_width_xaxis, marker_width_8_M0, color='axcolour', linestyle=axmarkers[0], linewidth=2, label='$L=8$')
# ax.plot(Marker_width_xaxis, marker_width_10_M0, color='k', linestyle=axmarkers[1], linewidth=2, label='$L=10$')
ax1.plot(Marker_width_xaxis, marker_width_12_M0, color=axcolour[0], marker='.', markersize=12, label='  $ \\nu$')
# ax.plot(Marker_width_xaxis, marker_width_8_M2, color='r', linestyle=axmarkers[0], linewidth=2, label='$L=8$')
# ax.plot(Marker_width_xaxis, marker_width_10_M2, color='r', linestyle=axmarkers[1], linewidth=2, label='$L=10$')
ax1.plot(Marker_width_xaxis, marker_width_12_M2, color=axcolour[1], marker='.', markersize=12, label='  $ \\nu$')

# Axis labels and limits
ax1.set_ylabel("$\\nu$", fontsize=35)
ax1.set_xlabel("$w$", fontsize=35)
ax1.set_xlim(0, 0.3)
ax1.set_ylim(-2.2, 2)
ax1.yaxis.set_label_coords(-0.075, 0.5)

# Axis ticks
ax1.tick_params(which='major', width=0.75)
ax1.tick_params(which='major', length=14)
ax1.tick_params(which='minor', width=0.75)
ax1.tick_params(which='minor', length=7)
majorsy = [-2, -1, 0, 1, 2]
minorsy = [-1.5, -0.5, 0.5, 1.5]
majorsx = [0, 0.1, 0.2, 0.3]
minorsx = [0.05, 0.15, 0.25]
ax1.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
ax1.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
ax1.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
ax1.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))

# Gap
right_ax = ax1.twinx()
# right_ax.plot(Marker_width_xaxis, gap_width_8_M0, color=axcolour[1], linestyle=axmarkers[0], linewidth=2)
# right_ax.plot(Marker_width_xaxis, gap_width_10_M0, color=axcolour[1], linestyle=axmarkers[1], linewidth=2)
right_ax.plot(Marker_width_xaxis, gap_width_12_M0, color=axcolour[0], linestyle=axmarkers[0], linewidth=2, label='$ \Delta $')
# right_ax.plot(Marker_width_xaxis, gap_width_8_M2, color=axcolour[1], linestyle=axmarkers[0], linewidth=2)
# right_ax.plot(Marker_width_xaxis, gap_width_10_M2, color=axcolour[1], linestyle=axmarkers[1], linewidth=2)
right_ax.plot(Marker_width_xaxis, gap_width_12_M2, color=axcolour[1], linestyle=axmarkers[0], linewidth=2, label='$ \Delta $')
right_ax.set_ylabel("$\Delta$", fontsize=35)
right_ax.set_ylim(-0.1, 2)

# Axis ticks
right_ax.tick_params(which='major', width=0.75)
right_ax.tick_params(which='major', length=14)
right_ax.tick_params(which='minor', width=0.75)
right_ax.tick_params(which='minor', length=7)
majorsy = [0, 0.5, 1, 1.5, 2]
minorsy = [0.25, 0.75, 1.25, 1.75]
right_ax.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
right_ax.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))

# Legend and inset text
ax1.legend(loc=(0.45, 0.8), ncol=2, columnspacing=1.0, frameon=False, fontsize=30)
right_ax.legend(loc=(0.45, 0.73), ncol=2, columnspacing=0.8, frameon=False, fontsize=30)
ax1.text(0.14, 1.6, "$M=0$", fontsize=30)
ax1.text(0.22, 1.6, "$M=2$", fontsize=30)

box = ax1.get_position()
box.x0 = box.x0 - 0.015
box.x1 = box.x1 - 0.015
ax1.set_position(box)


#%% Marker vs M

# Font format
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Main figure
#axcolour = ['#365365', '#71250E', '#6F6534'] #dark
axcolour = ['#FF416D', '#3F6CFF', '#00B5A1'] # light
ax2.plot(Marker_M_xaxis, marker_M_8, color=axcolour[0], marker='.', markersize=12, label='$L=$ $8$')
ax2.plot(Marker_M_xaxis, marker_M_10, color=axcolour[1], marker='.', markersize=12, label='$L=$ $10$')
ax2.plot(Marker_M_xaxis, marker_M_12, color=axcolour[2], marker='.', markersize=12, label='$L=$ $12$')

# Axis labels and limits
ax2.set_ylabel("$\\nu$", fontsize=35)
ax2.set_xlabel("$M$", fontsize=35)
ax2.set_xlim(-5, 5)
ax2.set_ylim(-2.2, 1.2)
ax2.yaxis.set_label_coords(-0.075, 0.5)

# Axis ticks
ax2.tick_params(which='major', width=0.75)
ax2.tick_params(which='major', length=14)
ax2.tick_params(which='minor', width=0.75)
ax2.tick_params(which='minor', length=7)
majorsy = [-2, -1, 0, 1]
minorsy = [ -1.5, -0.5, 0.5]
majorsx = [-5,  -3,  -1,  1, 3, 5]
minorsx = [-4, -2, 0, 2, 4]
ax2.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
ax2.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
ax2.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
ax2.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))

# Legend and inset text
ax2.legend(loc=(0.6, 0.24), frameon=False, fontsize=30)
# ax.text(-4.7, 0.92, "(a)", fontsize=20)


# Inset figure
# Placement and inset
# left, bottom, width, height = [0.72, 0.27, 0.24, 0.26]
left, bottom, width, height = [0.08, 0.23, 0.3, 0.3]
inset_ax = inset_axes(ax2, height="100%", width="100%", bbox_to_anchor = [left, bottom, width, height], bbox_transform = ax2.transAxes, loc = 3 ) # [left, bottom, width, height])
#insetcolour = ['#BF7F04', '#BF5B05', '#8C1C04'] # dark
insetcolour = ['#6668FF', '#FFC04D','#FF7D66'] # light

inset_ax.plot(M_inset_xaxis, M_inset_0, color=insetcolour[0], marker='.', markersize=8, label='$M=0$')
inset_ax.plot(M_inset_xaxis, M_inset_2, color=insetcolour[1], marker='.', markersize=8, label='$M=2$')
inset_ax.plot(M_inset_xaxis, M_inset_4, color=insetcolour[2], marker='.', markersize=8, label='$M=4$')
inset_ax.plot(M_inset_xaxis,np.repeat(1, 12), color=insetcolour[1], linestyle="dashed")
inset_ax.plot(M_inset_xaxis,np.repeat(-2, 12), color=insetcolour[0], linestyle="dashed")

# Axis labels and limits
inset_ax.set_ylabel("$\\nu$", fontsize=30)
inset_ax.set_xlabel("$n$", fontsize=30)
inset_ax.set_xlim(1, 12)
inset_ax.set_ylim(-3, 3)

# Axis labels
inset_ax.xaxis.set_label_coords(0.5, -0.05)
inset_ax.yaxis.set_label_coords(-0.13, 0.5)
# Axis ticks
inset_ax.tick_params(which='major', width=0.75, labelsize=15)
inset_ax.tick_params(which='major', length=7, labelsize=15)
inset_ax.tick_params(which='minor', width=0.75)
inset_ax.tick_params(which='minor', length=3.5)
majorsy2 = [-3, -1, 1, 3]
minorsy2 = [-2, 0, 2]
majorsx2 = [1, 12]
minorsx2 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
inset_ax.yaxis.set_major_locator(ticker.FixedLocator(majorsy2))
inset_ax.yaxis.set_minor_locator(ticker.FixedLocator(minorsy2))
inset_ax.xaxis.set_major_locator(ticker.FixedLocator(majorsx2))
inset_ax.xaxis.set_minor_locator(ticker.FixedLocator(minorsx2))

# Legend
inset_ax.text(1.1, 1.6, " $M=2$", fontsize=15)
inset_ax.text(1.1, -1.4, " $M=0$", fontsize=15)
inset_ax.text(1.1, 0.2, " $M=4$", fontsize=15)



#%% Phase diagram


# Font format
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Colormap
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
colormap1 = plt.get_cmap('coolwarm')
new_cmap = truncate_colormap(colormap1, 0, 0.75)


# Phase diagram
# fig, ax = plt.subplots(figsize=(8, 6))
scatters = ax3.scatter(pd_mesh_x, pd_mesh_y, c=phase_diagram, marker='s', cmap=new_cmap,  linewidths=2.5)
cbar = plt.colorbar(scatters, ax=ax3)
contours = ax3.contour(pd_gap_x, pd_gap_y, pd_gap.T, levels=[0.2], colors="black", linewidths=3.5)
# ax3.clabel(contours, inline=True, fontsize=22, inline_spacing=20, fmt="$\\Delta = 0.2 \hspace{2mm}$ ")

# Colorbar format
cbar.set_label(label='$\\nu$', size=35, labelpad=-15, y=0.5)
# cbar.set_label_coords(0.075, 0.5)
cbar.ax.tick_params(labelsize=25)
cbar.set_ticks([-2, -1, 0, 1, 2])
cbar.set_ticklabels([-2, -1, 0, 1, 2])

# Axis labels and limits
ax3.set_ylabel("$M$", fontsize=35)
ax3.set_xlabel("$w$", fontsize=35)
ax3.set_xlim(0, 0.3)
ax3.set_ylim(-3.5, 3.5)
ax3.yaxis.set_label_coords(-0.09, 0.5)

# Axis ticks
ax3.tick_params(which='major', width=0.75)
ax3.tick_params(which='major', length=14)
ax3.tick_params(which='minor', width=0.75)
ax3.tick_params(which='minor', length=7)
majorsy = [-3, -2, -1, 0, 1, 2, 3]
minorsy = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
majorsx = [0, 0.1, 0.2, 0.3]
minorsx = [0.05, 0.15, 0.25]
ax3.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
ax3.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
ax3.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
ax3.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))

# plt.tight_layout()
plt.savefig("try2.pdf", bbox_inches="tight")
plt.show()











