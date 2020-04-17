# this code was used for testing how many neurons from neuromorpho were compatible with TMD
import tmd
import os

directory = './granule_human/'

tot_files = 0
ok_files = 0

for file in os.listdir(directory):

    if file.endswith(".swc"):
        tot_files += 1
        try:
            # Load your morphology
            neu = tmd.io.load_neuron(directory + file)

        except:
            continue
        else:
            # plt.show(view.neuron(neu))
            ok_files += 1
            continue
    else:
        continue

print("tot " + str(tot_files))
print("ok " + str(ok_files))
