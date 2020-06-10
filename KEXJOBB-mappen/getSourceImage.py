import sys
import csv
import numpy as np
import matplotlib.cm as cm
from tmd.view import view, plot
import tmd
import matplotlib.pyplot as plt
from tmd.view.common import jet_map
import os


neuron = tmd.io.load_neuron('02a_pyramidal2aFI.CNG.swc')
pd = tmd.methods.get_persistence_diagram(neuron.neurites[0])


# Step 2: Load your morphology
filename = './granule_swc/03D23APV-1.CNG.swc'
neu = tmd.io.load_neuron(filename)

# Step 3: Extract the ph diagram of a tree
tree = neu.neurites[0]
ph = tmd.methods.get_persistence_diagram(tree)

# Step 4: Extract the ph diagram of a neuron's trees
ph_neu = tmd.methods.get_ph_neuron(neu)

# Step 5: Extract the ph diagram of a neuron's trees,
# depending on the neurite_type
ph_apical = tmd.methods.get_ph_neuron(neu, neurite_type='apical')
ph_axon = tmd.methods.get_ph_neuron(neu, neurite_type='axon')
ph_basal = tmd.methods.get_ph_neuron(neu, neurite_type='basal')

# Step 6: Plot the extracted topological data with three different ways


# Visualize the neuron
view.neuron(neu)

# Visualize a selected neurite type or multiple of them
# view.neuron(neu, neurite_type=['apical'])

# Visualize the persistence diagram
# plt.show(plot.diagram(ph_neu))

# Visualize the persistence barcode
plt.show(plot.barcode(ph_neu))
