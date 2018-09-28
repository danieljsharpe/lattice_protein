'''
Python script containing functions to implement local & global MC moves for lattice protein
'''

import numpy as np
from copy import deepcopy
from itertools import islice
from collections import deque

# A MoveError is raised if a proposed move results in a clash or if no valid moves can be proposed
class MoveError(Exception):
    pass

# Function to randomly propose a valid slither (reptation) move, if possible
def propose_slither_move(latt, coords, nbrs):
    slither_moves = zip(nbrs.tolist()*2,[0]*6 + [1]*6)
    while slither_moves:
        try:
            rn = np.random.randint(len(slither_moves)) # residue no.
            slither_move = slither_moves[rn]
            new_latt, new_coords = slither(latt, coords, slither_move[1], np.array(slither_move[0]))
        except MoveError: # the proposed slither move results in a clash
            del slither_moves[rn]
            continue
        else:
            break
    return new_latt, new_coords

def propose_pivot_move(latt, coords):
    Nres = len(coords)
    # choose an initial (internal) pivot residue, biased towards the centre of the chain
    pivot_res = int(np.random.randint(Nres)*np.random.normal(0.5,0.05))
    if pivot_res <= 1: pivot_res = 2
    elif pivot_res >= Nres-1: pivot_res = Nres-2
    search_dir = np.random.choice([-1,1])  # choose a search direction along the chain to find valid pivot moves
    while pivot_res >= 2 and pivot_res <= Nres-2:
        try:
            new_latt, new_coords = pivot(latt, deepcopy(coords), pivot_res, search_dir)
        except MoveError:
            pivot_res += search_dir
            if pivot_res < 2 or pivot_res > Nres-2: raise
            else: continue
        else:
            break
    return new_latt, new_coords

def mutate(latt, coords, mutatable_res):
    return

# slithering snake (reptation) global move
def slither(latt, coords, mv_direc, nbr):
    if mv_direc == 0: # from head
        new_ter = coords[0]+nbr
        if latt[new_ter[0],new_ter[1],new_ter[2]] != 0: raise MoveError
        coords.pop()
        coords.appendleft(new_ter.tolist())
    elif mv_direc == 1: # from tail
        new_ter = coords[-1]+nbr
        if latt[new_ter[0],new_ter[1],new_ter[2]] != 0: raise MoveError
        coords.popleft()
        coords.append(new_ter.tolist())
    latt = reconstruct_latt(coords)
    return latt, coords

# pivot global move about a chosen internal residue
def pivot(latt, coords, pivot_res, search_dir):
    pivot_res_orig = deepcopy(pivot_res)
    Nres = len(coords)
    pivot_ax = np.array(coords[pivot_res]) - np.array(coords[pivot_res+search_dir])
    pivot_ax_idcs = np.where(pivot_ax==0)[0]
    pivot_move = np.random.randint(3)
    while pivot_res >= 0 and pivot_res <= Nres-1:
        if pivot_move == 0 or pivot_move == 2: # quarter-turn & three-quarter-turn, respectively
            delta_1 = np.array(coords[pivot_res][pivot_ax_idcs[1]]) - np.array(coords[pivot_res_orig][pivot_ax_idcs[1]])
            delta_2 = np.array(coords[pivot_res][pivot_ax_idcs[0]]) - np.array(coords[pivot_res_orig][pivot_ax_idcs[0]])
            if pivot_move == 0:
                coords[pivot_res][pivot_ax_idcs[0]] = coords[pivot_res_orig][pivot_ax_idcs[0]] - delta_1
                coords[pivot_res][pivot_ax_idcs[1]] = coords[pivot_res_orig][pivot_ax_idcs[1]] - delta_2
            elif pivot_move == 2:
                coords[pivot_res][pivot_ax_idcs[0]] = coords[pivot_res_orig][pivot_ax_idcs[0]] + delta_1
                coords[pivot_res][pivot_ax_idcs[1]] = coords[pivot_res_orig][pivot_ax_idcs[1]] + delta_2
        elif pivot_move == 1: # half-turn
            for i in range(len(pivot_ax_idcs)):
                coords[pivot_res][pivot_ax_idcs[i]] -= 2*(coords[pivot_res][pivot_ax_idcs[i]] - \
                        coords[pivot_res_orig][pivot_ax_idcs[i]])
        pivot_res += search_dir
    if search_dir == -1:
        stat_segment = deque(islice(coords,pivot_res_orig,Nres)) # segment of protein that stays put
    elif search_dir == +1:
        stat_segment = deque(islice(coords,0,pivot_res_orig))
    pivot_res = deepcopy(pivot_res_orig)
    for i in range(Nres-pivot_res_orig-1):
        if coords[pivot_res] in stat_segment:
            raise MoveError # there is a clash
        pivot_res += search_dir
    latt = reconstruct_latt(coords)
    return latt, coords

# function to perform local kink-jump move of internal residue
def kink_jump(latt, coords):
    nbr_config = []
    for i in range(1,len(coords)-1):
        nbr_config.append(np.array(coords[i+1]) - np.array(coords[i-1]))
    config = nbr_config[2]
    # for a kink-jump move to be possible, the residue must be at a corner
    kink_jump_sites = [i+1 for i, config in enumerate(nbr_config) if not np.where(config==2)[0]]
    if not kink_jump_sites: raise MoveError
    while kink_jump_sites:
        kink_jump_site_idx = np.random.randint(len(kink_jump_sites))
        kink_jump_res = kink_jump_sites[kink_jump_site_idx]
        coords_1 = np.array(coords[kink_jump_res+1]) - np.array(coords[kink_jump_res])
        coords_2 = np.array(coords[kink_jump_res-1]) - np.array(coords[kink_jump_res])
        new_coords = coords_1 + coords_2 + np.array(coords[kink_jump_res])
        if latt[new_coords[0],new_coords[1],new_coords[2]] != 0:
            del kink_jump_sites[kink_jump_site_idx]
            if not kink_jump_sites: raise MoveError
            else: continue
        else:
            coords[kink_jump_res] = new_coords.tolist()
            break
    latt = reconstruct_latt(coords)
    return latt, coords

# function to perform local crankshaft move
def crankshaft(latt, coords):
    crank_sites = []
    move_success = False
    for i in range(len(coords)-3):
        for j in range(i+3,len(coords)):
            if np.sum(abs(x) for x in (np.array(coords[i]) - np.array(coords[j]))) == 1: crank_sites.append((i,j))
    if not crank_sites: raise MoveError
    while crank_sites:
        crank_sites_idx = np.random.randint(len(crank_sites))
        crank_axis = np.array(coords[crank_sites[crank_sites_idx][0]]) - np.array(coords[crank_sites[crank_sites_idx][1]])
        crank_ax_idcs = np.where(crank_axis==0)[0]
        crank_moves = [0,1,2]
        while crank_moves:
            crank_move_idx = np.random.randint(len(crank_moves))
            crank_move = crank_moves[crank_move_idx]
            # section of protein chain that remains stationary (either side of residues forming the crank)
            stat_segment = deque(list(islice(coords,0,crank_sites[crank_sites_idx][0])) + \
                                 list(islice(coords,crank_sites[crank_sites_idx][1]+1,len(coords))))
            crank_segment = deque(islice(coords,crank_sites[crank_sites_idx][0]+1,crank_sites[crank_sites_idx][1]))
            new_crank_segment = deepcopy(crank_segment)
            try:
                res = 0
                for res_coord in crank_segment:
                    if crank_move == 0 or crank_move == 2:
                        delta_1 = np.array(res_coord[crank_ax_idcs[1]]) - \
                                  np.array(coords[crank_sites[crank_sites_idx][0]][crank_ax_idcs[1]])
                        delta_2 = np.array(res_coord[crank_ax_idcs[0]]) - \
                                  np.array(coords[crank_sites[crank_sites_idx][0]][crank_ax_idcs[0]])
                        if crank_move == 0: # quarter-turn
                            new_crank_segment[res][crank_ax_idcs[0]] = coords[crank_sites[crank_sites_idx][0]][crank_ax_idcs[0]] \
                                    - delta_1
                            new_crank_segment[res][crank_ax_idcs[1]] = coords[crank_sites[crank_sites_idx][0]][crank_ax_idcs[1]] \
                                    - delta_2
                        elif crank_move == 2: # three-quarter-turn
                            new_crank_segment[res][crank_ax_idcs[0]] = coords[crank_sites[crank_sites_idx][0]][crank_ax_idcs[0]] \
                                    + delta_1
                            new_crank_segment[res][crank_ax_idcs[1]] = coords[crank_sites[crank_sites_idx][0]][crank_ax_idcs[1]] \
                                    + delta_2
                    elif crank_move == 1: # half-turn
                        for i in range(len(crank_ax_idcs)):
                            new_crank_segment[res][crank_ax_idcs[i]] -= 2*(new_crank_segment[res][crank_ax_idcs[i]] - \
                                            coords[crank_sites[crank_sites_idx][0]][crank_ax_idcs[i]])
                    if new_crank_segment[res] in stat_segment:
                        raise MoveError # there is a clash
                    res += 1
                move_success = True
                break
            except MoveError:
                del crank_moves[crank_move_idx]
                continue
        if move_success: break
    if not crank_sites: raise MoveError
    j = 0
    for i in range(crank_sites[crank_sites_idx][0]+1,crank_sites[crank_sites_idx][1]):
        coords[i] = new_crank_segment[j]
        j += 1
    latt = reconstruct_latt(coords)
    return latt, coords

def pull(latt, coords):
    latt = reconstruct_latt(coords)
    return latt, coords

# function to reverse the protein sequence
def reverse(latt, coords):
    coords = deque([coords[-(i+1)] for i in range(len(coords))])
    latt = reconstruct_latt(coords)
    return latt, coords

# function to reconstruct lattice from the coords
def reconstruct_latt(coords):
    latt = np.zeros(shape=[int(float(len(coords))*1.5)]*3,dtype=int)
    for i in range(len(coords)):
        latt[coords[i][0],coords[i][1],coords[i][2]] = i+1
    return latt
