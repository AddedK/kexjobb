# Example to extract the persistence diagram from a neuronal tree

# Step 1: Import the tmd module
import tmd
from tmd.view import view, plot
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
import time as time

type_name = "granule"
saved_X_filename = type_name + '_X_train.txt'

X = np.loadtxt(saved_X_filename)


st = time.time()
ward_labels = AgglomerativeClustering(n_clusters=8, linkage='ward').fit_predict(X)
elapsed_time = time.time() - st


print("Elapsed time: %.2fs" % elapsed_time)
print("Number of points: %i" % ward_labels.size)
print(ward_labels)

st = time.time()
ap_labels = AffinityPropagation().fit_predict(X)
elapsed_time = time.time() - st

print("Elapsed time: %.2fs" % elapsed_time)
print("Number of points: %i" % ap_labels.size)
print(ap_labels)

"""
gå igenom directory/typen
    de första x kolumnerna i dataset matrisen kommer tillhöra typen
    vi fyller label arrayen, de första x indexen med denna typ
    
"""
            

"""

directory = './purkinje/'

for file in os.listdir(directory):

    if file.endswith(".swc") :
        #Load your morphology
        neu = tmd.io.load_neuron(directory + file)
        plt.show(view.neuron(neu))
        continue
    else:
        continue
        
        """

"""

 # Step 2: Load your morphology
filename = '02a_pyramidal2aFI.CNG.swc'
neu = tmd.io.load_neuron(filename)

# Step 3: Extract the ph diagram of a tree
tree = neu.neurites[0]
ph = tmd.methods.get_persistence_diagram(tree)

# Step 4: Extract the ph diagram of a neuron's trees
ph_neu = tmd.methods.get_ph_neuron(neu)

print(ph_neu)

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

# Visualize a selected neurite type or multiple of them
plt.show(view.neuron(neu, neurite_type=['apical']))

# Visualize the persistence diagram
plt.show(plot.diagram(ph_apical))

# Visualize the persistence barcode
plt.show(plot.barcode(ph_apical))

# Visualize the persistence image
plt.show(plot.persistence_image(ph_neu))

"""
