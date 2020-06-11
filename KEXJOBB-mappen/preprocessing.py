import tmd
import numpy as np
import os

"""
SCREENING FOR SWC FILES THAT WORK WITH THE TMD ALGORITHM
ONLY ACCEPTS COMPLETE NEURON RECONSTRUCTIONS 
PERSISTENCE IMAGES ARE CREATED FOR THE SWC FILES THAT PASS THE SCREENING
VALUES FOR THE PERSISTENCE IMAGES ARE STORED ROW-WISE IN A SEPARATE FILE "pyramidal_X_train.txt"
E.G FOR PYRAMIDAL NEURON TYPE
"""


type_name = "pyramidal"
directory = './pyramidal/'
saved_X_filename = type_name + '_X_train.txt'

X = []

i = 0
for file in os.listdir(directory):
    if file.endswith(".swc"):
        try:
            # Load your morphology
            neu = tmd.io.load_neuron(directory + file)
            persistence_diagram = tmd.methods.get_ph_neuron(neu)
            persistence_image = tmd.analysis.get_persistence_image_data(
                persistence_diagram)
            i += 1
        except:
            continue
        else:

            persistence_image_vector = persistence_image.flatten()
            X.append(persistence_image_vector)
            continue
    else:
        continue

print("Number of neurons that got through persistence image test : " + str(i))
np.savetxt(saved_X_filename, X)

