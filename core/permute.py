import numpy as np


# TODO: Insert this into the permute_P function

# rearranges the machine precedence adjacency matrix (B) with permutation of machine ordering
# returns the rearranged B (does not change input)
def permute_B(B, permutation):
    M = B.shape[0]
    new_B = np.zeros(B.shape)
    for i in range(M):
        for j in range(M):
            new_B[i, j] = B[permutation[i], permutation[j]]
    return new_B

# rearranges the sample of the completion time function (sample) using a reordering (permutation) of the machines
# returns the rearranged sample (does not change input)
def permute_sample(sample, permutation):
    new_sample = np.zeros(sample.shape)
    for i in range(sample.shape[1]):
        new_sample[:, i, :] = sample[:, permutation[i], :]
    return new_sample

# rearranges machine precedence matrix adjacency matrix (B) with a subset of machines, representing a particular type
# type_machines is a vector of the machines in the original ordering of that type
# returns the rearranged B (does not change input)
def permute_typed_B(B, permutation, type_machines):
    num_type_machines = len(type_machines)
    new_B = B.copy()
    for i in range(num_type_machines):
        for j in range(num_type_machines):
            new_B[type_machines[i], type_machines[j]] = B[permutation[i], permutation[j]]
    return new_B

# rearranges a sample of the completion time function with a subset of machines, representing a particular type
# type_machines is a vector of the machines in the original ordering of that type
# returns the rearranged sample (does not change input)
def permute_typed_sample(sample, permutation, type_machines):
    # Copy because it is only a partial permutation
    new_sample = sample.copy()
    num_type_machines = len(type_machines)
    for i in range(num_type_machines):
        new_sample[:, type_machines[i], :] = sample[:, permutation[i], :]
    return new_sample
