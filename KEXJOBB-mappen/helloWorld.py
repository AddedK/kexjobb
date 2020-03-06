# Example to extract the persistence diagram from a neuronal tree

# Step 1: Import the tmd module
import tmd
from tmd.view import view, plot
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Load your morphology
filename = '02a_pyramidal2aFI.CNG.swc'
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
# Added kommentar: Vet inte om det är rätt men:
# Man behöver köra plt.show(...) på funktionen: t.ex view.neuron kallar inte plt.show
plt.show(view.neuron(neu))
