import sys
import csv
import numpy as np
import matplotlib.cm as cm
from tmd.view import view, plot
import tmd
import matplotlib.pyplot as plt
from tmd.view.common import jet_map
import os


neu = tmd.io.load_neuron('02a_pyramidal2aFI.CNG.swc')
ph = tmd.methods.get_ph_neuron(neu)
# result = plot.persistence_image(ph)
plt.show(view.neuron(neu))
# plt.imsave("pyramidalPersistenceExample.png",
#            result, cmap=jet_map)
# plt.show("pyramidalPersistenceExample.png", result)
