
# Import the TMD toolkit in IPython
import tmd

# Load a neuron
neuron = tmd.io.load_neuron('../tests/data/valid/C010398B-P2.CNG.swc')

# Extract the tmd of a neurite, i.e., neuronal tree
pd = tmd.methods.get_persistence_diagram(neuron.neurites[0])

print(pd)
