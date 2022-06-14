# Collecting data and making plots for the marker against M


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import h5py


Ls =[8, 10, 12]  # System sizes
Ms = [2]  # Mass parameter
Ws = np.linspace(0, 0.2, 50)
ts = 0.5  # Chiral symmetry breaking
Rs = np.arange(4)  # Realisations
Ws_string = []

with open('width_values.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
        params = line.split()
        Ws_string.append(params[0])

local_marker0 = np.zeros((len(Ws), len(Rs)))
local_marker1 = np.zeros((len(Ws), len(Rs)))
local_marker2 = np.zeros((len(Ws), len(Rs)))


for file in os.listdir('outdir_width'):

    if file.endswith('h5'):

        file_path = os.path.join("outdir_width", file)

        with h5py.File(file_path, 'r') as f:

            datanode = f['data']
            value = datanode[()]
            size = datanode.attrs['size']
            M = datanode.attrs['M']
            width = datanode.attrs['width']
            w_string = '{:.8f}'.format(width)
            R = datanode.attrs['R']
            row, col = Ws_string.index(w_string), R

            if size == 8:
                local_marker0[row, col] = value
            elif size == 10:
                local_marker1[row, col] = value
            elif size == 12:
                local_marker2[row, col] = value






marker0 = np.mean(local_marker0, axis=1)
marker1 = np.mean(local_marker1, axis=1)
print(marker1)
marker2 = np.mean(local_marker2, axis=1)
print("---------")
print(marker2)


std0 = np.std(local_marker0, axis=1) / np.sqrt(len(Rs))
std1 = np.std(local_marker1, axis=1) / np.sqrt(len(Rs))
std2 = np.std(local_marker2, axis=1) / np.sqrt(len(Rs))



# %% Plots


# Figure 1
# Font format
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Main figure
fig, ax = plt.subplots(figsize=(8, 6))
axcolour = ['#365365', '#71250E', '#6F6534', '#9932CC']
ax.plot(Ws, marker0, color=axcolour[0], marker='.', markersize=12, label='$L= 8$' + " $t_2=0.5$")
ax.plot(Ws, marker1, color=axcolour[1], marker='.', markersize=12, label='$L= 10$' + " $t_2=0.5$")
ax.plot(Ws, marker2, color=axcolour[2], marker='.', markersize=12, label='$L= 12$' + " $t_2=0.5$")
ax.plot(Ws, np.repeat(0, len(Ws)), '--')
ax.plot(Ws, np.repeat(1, len(Ws)), '--')

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
plt.show()
plt.savefig("try1.pdf", bbox_inches="tight")
plt.savefig("try1.eps", bbox_inches="tight")

print("done")