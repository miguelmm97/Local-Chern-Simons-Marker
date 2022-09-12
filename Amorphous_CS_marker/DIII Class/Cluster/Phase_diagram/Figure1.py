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
            phase_diagram= datanode

    if file == "Phase_diagram_gap_results.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            pd_gap= datanode

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
            Marker_M_xaxis = datanode

    if file == "Marker_width_Xaxis_results.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            Marker_width_xaxis = datanode

    if file == "Phase_diagram_mesh_results.h5":
        with h5py.File(file, 'r') as f:
            datanode = f['data']
            pd_mesh = datanode