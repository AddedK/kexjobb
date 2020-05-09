import numpy as np
import fileinput


nr_of_labels = [109, 452, 856, 459, 57, 1875]
label_names = ["glutamatergic", "granule",
               "medium_spiny", "basket", "fast_spiking", "pyramidal"]

X_train_list = ["glutamatergic_X_train.txt", "granule_X_train.txt", "medium_spiny_X_train.txt",
                "basket_X_train.txt", "fast_spiking_X_train.txt", "pyramidal_X_train.txt"]

#labels  = np.zeros(sum(nr_of_labels))
labels = []


def create_labels():

    for label_amount, label_name in zip(nr_of_labels, label_names):
        for i in range(label_amount):
            labels.append(label_name)

    np.savetxt("labels.txt", labels,  fmt='%s')


def create_dataset():
    with open('X_train_all_types.txt', 'w') as file:
        input_lines = fileinput.input(X_train_list)
        file.writelines(input_lines)


create_labels()
# print(labels)
create_dataset()
