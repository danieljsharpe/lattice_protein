'''
Python script containing functions for calculating thermodynamic quantities and order parameters for lattice protein
'''

import numpy as np

def calc_energy(protein, latt, coords, res_dict, score_mtx, nbrs):
    energy = 0.0
    for i in range(len(protein)):
        curr_cell = coords[i]
        # check neighbours
        for nbr in nbrs:
            nbr_res = latt[curr_cell[0]+nbr[0],curr_cell[1]+nbr[1],curr_cell[2]+nbr[2]]
            if nbr_res != 0 and nbr_res != i and nbr_res != i+2: # cell is not empty and is not an adjcent res in the chain
#                print "working res:", i+1, protein[i], "nbr res:", nbr_res, protein[nbr_res-1]
                energy += 0.5*score_mtx[res_dict[protein[i]],res_dict[protein[nbr_res-1]]]
    return energy

# Calculate partition function
def calc_part_func():
    return

# Count number of native contacts in a configuration (order parameter "Q")
def calc_q(protein, latt, coords, nbrs, native_contacts):
    Q = 0
    return Q

# Count number of contacts in a configuration (order parameter "K")
def calc_k_nbrs(protein, latt, coords, nbrs):
    K = 0
    for i in range(len(protein)):
        curr_cell = coords[i]
        for nbr in nbrs:
            nbr_res = latt[curr_cell[0]+nbr[0],curr_cell[1]+nbr[1],curr_cell[2]+nbr[2]]
            if nbr_res != 0 and nbr_res != i and nbr_res != i+2 and nbr_res > i+1:
                K += 1
    return K

# Calculate end-to-end distance (order parameter "R")
def calc_endtoend(coords):
    R = 1.
    return R
