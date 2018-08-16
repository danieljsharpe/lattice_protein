'''
Python scripts for simulating the folding of a lattice protein
Daniel J. Sharpe
2018
'''

import numpy as np
from copy import deepcopy
from collections import deque
import thermo_func
import mcmove
import visualise_protein

class Lattice_Protein(object):

    # Indices for hydrophobic, polar, anionic and cationic residues in HP scoring matrix
    HP_dict = {"H": 0, "P": 1, "A": 2, "C": 3}
    HP_score = np.array([
            [ 0.03, 0.06, 0.02, -0.03 ], [ 0.06,  0.00, 0.38, -0,76], \
            [ 0.02, 0.38, 0.02,  0.38 ], [-0.03, -0.76, 0.38,  0.12]
            ])

    # Indices for all amino acids in MJ & BT scoring matrices
    allres_dict = {"C": 0, "F": 1, "L": 2, "W": 3, "V": 4, "I": 5, "M": 6, "H": 7, "Y": 8, \
                   "A": 9, "G": 10, "P": 11, "N": 12, "T": 13, "S": 14, "R": 15, "Q": 16, \
                   "N": 17, "K": 18, "E": 19}
    MJ_score = []

    BT_score = np.array([
            [-1.34,-0.53,-0.50,-0.74,-0.51,-0.48,-0.49,-0.19,-0.16,-0.26,-0.09,-0.18, 0.28, 0.00, 0.09, 0.32, 0.04, 0.38, 0.35, 0.46], \
            [-0.53,-0.82,-0.78,-0.78,-0.67,-0.65,-0.89,-0.19,-0.49,-0.33, 0.11,-0.19, 0.29, 0.00, 0.10, 0.08,-0.04, 0.48, 0.11, 0.34], \
            [-0.50,-0.78,-0.81,-0.70,-0.80,-0.79,-0.68, 0.10,-0.44,-0.37, 0.14,-0.08, 0.36, 0.00, 0.26, 0.09, 0.08, 0.62, 0.16, 0.37], \
            [-0.74,-0.78,-0.70,-0.74,-0.62,-0.65,-0.94,-0.46,-0.55,-0.40,-0.24,-0.73,-0.09, 0.00, 0.07,-0.41,-0.11, 0.06,-0.28,-0.15], \
            [-0.51,-0.67,-0.80,-0.62,-0.72,-0.68,-0.47, 0.18,-0.27,-0.38, 0.04,-0.08, 0.39, 0.00, 0.25, 0.17, 0.17, 0.66, 0.16, 0.41], \
            [-0.48,-0.65,-0.79,-0.65,-0.68,-0.60,-0.60, 0.19,-0.33,-0.35, 0.21, 0.05, 0.55, 0.00, 0.35, 0.18, 0.14, 0.54, 0.21, 0.38], \
            [-0.49,-0.89,-0.68,-0.94,-0.47,-0.60,-0.56,-0.17,-0.51,-0.23, 0.08,-0.16, 0.32, 0.00, 0.32, 0.17,-0.01, 0.62, 0.22, 0.24], \
            [-0.19,-0.19, 0.10,-0.46, 0.18, 0.19,-0.17,-0.33,-0.21, 0.21, 0.23,-0.05, 0.10, 0.00, 0.15, 0.04, 0.22,-0.22, 0.26,-0.11], \
            [-0.16,-0.49,-0.44,-0.55,-0.27,-0.33,-0.51,-0.21,-0.27,-0.15,-0.04,-0.40, 0.01, 0.00, 0.07,-0.37,-0.18,-0.07,-0.40,-0.16], \
            [-0.26,-0.33,-0.37,-0.40,-0.38,-0.35,-0.23, 0.21,-0.15,-0.20,-0.03, 0.07, 0.24, 0.00, 0.15, 0.27, 0.21, 0.30, 0.20, 0.43], \
            [-0.09, 0.11, 0.14,-0.24, 0.04, 0.21, 0.08, 0.23,-0.04,-0.03,-0.20,-0.01, 0.10, 0.00, 0.10, 0.14, 0.20, 0.17, 0.12, 0.48], \
            [-0.18,-0.19,-0.08,-0.73,-0.08, 0.05,-0.16,-0.05,-0.40, 0.07,-0.01,-0.07, 0.13, 0.00, 0.17,-0.02,-0.05, 0.25, 0.12, 0.26], \
            [ 0.28, 0.29, 0.36,-0.09, 0.39, 0.55, 0.32, 0.10, 0.01, 0.24, 0.10, 0.13,-0.04, 0.00, 0.14, 0.02,-0.05,-0.12,-0.14,-0.01], \
            [ 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], \
            [ 0.00, 0.10, 0.26, 0.07, 0.25, 0.35, 0.32, 0.15, 0.07, 0.15, 0.10, 0.17, 0.14, 0.00, 0.13, 0.12, 0.25, 0.01, 0.10, 0.10], \
            [ 0.32, 0.08, 0.09,-0.41, 0.17, 0.18, 0.17, 0.04,-0.37, 0.27, 0.14,-0.02, 0.02, 0.00, 0.12, 0.13,-0.12,-0.71, 0.50,-0.75], \
            [ 0.04,-0.04, 0.08,-0.11, 0.17, 0.14,-0.01, 0.22,-0.18, 0.21, 0.20,-0.05,-0.05, 0.00, 0.25,-0.12, 0.14, 0.12,-0.20, 0.10], \
            [ 0.38, 0.48, 0.62, 0.06, 0.66, 0.54, 0.62,-0.22,-0.07, 0.30, 0.17, 0.25,-0.12, 0.00, 0.01,-0.71, 0.12, 0.27,-0.69, 0.40], \
            [ 0.35, 0.11, 0.16,-0.28, 0.16, 0.21, 0.22, 0.26,-0.40, 0.20, 0.12, 0.12,-0.14, 0.00, 0.10, 0.50,-0.20,-0.69, 0.38,-0.87], \
            [ 0.46, 0.34, 0.37,-0.15, 0.41, 0.38, 0.24,-0.11,-0.16, 0.43, 0.48, 0.26,-0.01, 0.00, 0.10,-0.75, 0.10, 0.40,-0.87, 0.45]
           ])

    nbrs = np.array([[-1,0,0],[+1,0,0],[0,-1,0],[0,+1,0],[0,0,-1],[0,0,+1]], dtype=int) # neighbours in the six cardinal directions in 3D

    def __init__(self, protein=None, go_score=None, randseed=17, potential="bt"):
        self.protein = protein # amino acid sequence as a string
        self.Nres = len(protein) # no. of residues in the protein
        if potential == "hp":
            self.scores = Lattice_Protein.HP_score
            self.res_dict = Lattice_Protein.HP_dict
            for i in range(self.Nres): assert self.protein[i] in self.res_dict
        elif potential == "bt":
            self.scores = Lattice_Protein.BT_score
            self.res_dict = Lattice_Protein.allres_dict
            for i in range(self.Nres): assert self.protein[i] in self.res_dict
        elif potential == "go":
            self.scores = go_score
            if go_score is None: raise ValueError("If using the Go potential must provide a score matrix go_score")
            for i in range(self.Nres): assert self.protein[i] is int
        np.random.seed(randseed)
        self.latt = None
        while self.latt is None: # keep trying to initialise lattice until successful
            self.latt, self.coords = self.init_latt()

    # Initialise a random configuration on a lattice by chain growth
    def init_latt(self):
        latt = np.zeros(shape=[int(float(self.Nres)*1.5)]*3,dtype=int)
        curr_cell = [self.Nres/2]*3
        latt[curr_cell[0],curr_cell[1],curr_cell[2]] = 1 # place head res of protein
        coords = deque([[curr_cell[0],curr_cell[1],curr_cell[2]]])
        for i in range(2,self.Nres+1):
            avail_nbrs = deepcopy(Lattice_Protein.nbrs) # available neighbours
            prev_cell = deepcopy(curr_cell)
            while avail_nbrs.any:
                nbr = avail_nbrs[np.random.randint(np.shape(avail_nbrs)[0])]
                curr_cell += nbr
                if latt[curr_cell[0],curr_cell[1],curr_cell[2]] == 0:
                    latt[curr_cell[0],curr_cell[1],curr_cell[2]] = i
                    coords.append([curr_cell[0],curr_cell[1],curr_cell[2]])
                    break
                else:
                    avail_nbrs = np.delete(avail_nbrs, np.where((avail_nbrs[:,0]==nbr[0]) & (avail_nbrs[:,1]==nbr[1]) \
                                 & (avail_nbrs[:,2]==nbr[2]))[0][0], 0)
                    curr_cell = deepcopy(prev_cell)
            if not avail_nbrs.any:
                return None # got stuck trying to generate a configuration (no available neighbours)
            else:
                continue
        return latt, coords

class Simulate_Protein(Lattice_Protein):

    '''
    Initialise a MC walker. Note: the arguments to the Lattice_Protein class are passed as kwargs to the Simulate_Protein subclass
    '''
    def __init__(self, T_init, n_steps, glob_mv_freq=5, global_move_prob=(0.4,0.55,0.05), local_move_prob=(0.9,0.1), \
                 annealing=True, sa_steps=5000, alpha=0.995, schedule="exponential", p_acc_thresh=0.05, p_acc_thresh_nsteps=100,  \
                 T_thresh=0.001, *args, **kwargs):
        super(Simulate_Protein, self).__init__(*args, **kwargs)
        self.T = T_init # current temperature and initial temperature, respectively
        self.n_steps = n_steps # total number of MC steps in simulation run
        self.glob_mv_freq = glob_mv_freq # step intervals at which global moves are attempted
        self.global_move_prob = global_move_prob # probabilities for attempting reptation, pivot & reverse global moves, respectively
        self.local_move_prob = local_move_prob # probabilities for attempting kink-jump & crankshaft local moves, respectively
        self.annealing = annealing # after MC simulation run simulated annealing Y/N
        if self.annealing: # set a method and parameters for updating the temperature
            self.sa_steps = sa_steps # max. no. of steps in simulated annealing procedure
            self.alpha = alpha # cooling coeff for simulated annealing procedure
            self.p_acc_thresh = p_acc_thresh # set an acceptance prob. cutoff for terminating the algorithm
            self.p_acc_thresh_nsteps = p_acc_thresh_nsteps # no. of steps defining a block for which the acceptance prob. is calcd
            self.T_thresh = T_thresh # set a temperature cutoff for terminating the algorithm
            if schedule == "exponential": self.cooling_func = Simulate_Protein.cooling_exponential
            elif schedule == "linear": self.cooling_func = Simulate_Protein.cooling_linear
            elif schedule == "logarithmic": self.cooling_func = Simulate_Protein.cooling_logarithmic
        return

    '''
    Function to drive Monte Carlo simulation for parallel tempering or simulated annealing procedure
    '''
    def monte_carlo(self, annealing=False):
        if annealing: self.n_steps = self.sa_steps
        n_step = 0
        p_accept = 1.0
        n_accept = 0
        E_old = thermo_func.calc_energy(self.protein,self.latt,self.coords,self.res_dict,self.scores,Lattice_Protein.nbrs)
        while n_step < self.n_steps and self.T > self.T_thresh and p_accept > self.p_acc_thresh:
            if n_step % self.glob_mv_freq == 0: # make global move
                move_type = np.random.choice(("slither","pivot","reverse"),p=self.global_move_prob)
                if move_type == "slither":
                    new_latt, new_coords = mcmove.propose_slither_move(self.latt, deepcopy(self.coords), Lattice_Protein.nbrs)
                elif move_type == "pivot":
                    try:
                        new_latt, new_coords = mcmove.propose_pivot_move(self.latt, deepcopy(self.coords))
                    except mcmove.MoveError:
                        continue
                elif move_type == "reverse":
                    new_latt, new_coords = mcmove.reverse(self.latt, deepcopy(self.coords))
            else: # make local move
                move_type = np.random.choice(("kinkjump","crankshaft"),p=self.local_move_prob)
                if move_type == "kinkjump":
                    try:
                        new_latt, new_coords = mcmove.kink_jump(self.latt, deepcopy(self.coords))
                    except mcmove.MoveError:
                        continue
                elif move_type == "crankshaft":
                    try:
                        new_latt, new_coords = mcmove.crankshaft(self.latt, deepcopy(self.coords))
                    except mcmove.MoveError:
                        continue
            E_new = thermo_func.calc_energy(self.protein,new_latt,new_coords,self.res_dict,self.scores,Lattice_Protein.nbrs)
#            print "E_old:", E_old, "E_new:", E_new
            if Simulate_Protein.metropolis(Simulate_Protein.accept_prob(E_old,E_new,self.T)):
                n_accept += 1
                E_old = E_new
                self.latt, self.coords = deepcopy(new_latt), deepcopy(new_coords)
            else:
                pass
            n_step += 1
            if n_step % self.p_acc_thresh_nsteps == 0: # check acceptance probability for block
                p_accept = float(n_accept)/float(self.p_acc_thresh_nsteps)
                p_accept = 1.0 # DEBUG - don't let the loop break because of p_accept
                n_accept = 0 # reset
            if annealing:
                self.T = self.cooling_func(self.T, self.alpha)
#                print "Temperature is now:", self.T
#        print "coords at end of simulation:\n", self.coords
        print "number of steps taken in block:", n_step
        print "energy at end of block:", E_old
#        visualise_protein.plot_protein(self.protein,self.coords)

    # Metropolis criterion
    @staticmethod
    def metropolis(p_acc):
        if np.random.rand() < p_acc:
            return True
        else:
            return False

    # Metropolis acceptance probability
    @staticmethod
    def accept_prob(E_old, E_new, T):
        return np.exp(-(E_new - E_old)/T)

    @staticmethod
    def cooling_exponential(T, alpha):
        return T*alpha

    @staticmethod
    def cooling_linear(T, alpha):
        return T-alpha

    @staticmethod
    def cooling_logarithmic(T_0, n_step):
        return T_0 / np.log(1.+n_step)

    '''
    Trade temperatures of two (adjacent) walkers / replicas (i.e. parallel tempering move)
    walker1 & walker2 are two replicas (instances of the Simulate_Protein subclass)
    '''
    @classmethod
    def replica_exchange_move(cls, walker1, walker2):
        E_walker1 = thermo_func.calc_energy(walker1.protein,walker1.latt,walker1.coords,walker1.res_dict,walker1.scores, \
                                            Lattice_Protein.nbrs)
        E_walker2 = thermo_func.calc_energy(walker2.protein,walker2.latt,walker2.coords,walker2.res_dict,walker2.scores, \
                                            Lattice_Protein.nbrs)
        if cls.metropolis(cls.replica_exchange_accept_prob(E_walker1,E_walker2,walker1.T,walker2.T)):
            walker2_temp = walker2.T
            walker2.T = walker1.T
            walker1.T = walker2_temp
            return True
        else:
            return False

    @staticmethod
    def replica_exchange_accept_prob(E_walker1, E_walker2, T_walker1, T_walker2):
        return np.exp(-((1./T_walker1)-(1./T_walker2))*(E_walker2-E_walker1))

'''
Set up the MC walkers and drive the simulation.
Note: the arguments to the Simulate_Protein subclass are passed as args
      the arguments to the Simulate_Protein subclass needed by the Lattice_Protein superclass are passed as kwargs
'''
def drive_simulation(N_walkers, T_min, T_max, tot_nsteps, exchange_interval, *args, **kwargs):
    delta_T = (T_max - T_min) / float(N_walkers-1)
    print "Setting up %i MC walkers with T_min = %.3f, T_max = %.3f, delta_t = %.3f" % (N_walkers,T_min,T_max,delta_T)
    print "Total number of MC steps: %i   Steps in a single block before replica exchange attempt: %i" \
          % (tot_nsteps, exchange_interval)
    print "Simulation parameters:"
    for kw, kwarg in kwargs.iteritems():
        print kw, "\t", kwarg
    temps = [T_min + (i*delta_T) for i in range(N_walkers)]
    temp_order = [i for i in range(N_walkers)] # keep track of which walkers are adjacent in temp
    walkers = [Simulate_Protein(temps[i],exchange_interval,*args,**kwargs) for i in range(N_walkers)]
    n_blocks = tot_nsteps / exchange_interval
    for block in range(n_blocks+1):
        print "BLOCK %i" % (block+1)
        for i, walker in enumerate(walkers):
            print "WALKER %i" % (i+1)
            if block < n_blocks:
                walker.monte_carlo()
            elif walker.annealing and block == n_blocks:
                walker.monte_carlo(annealing=True)
                visualise_protein.plot_protein(walker.protein,walker.coords)
            elif not walker.annealing and block == n_blocks:
                break
        if block == n_blocks: break
        while True:
            walker_idx = np.random.randint(N_walkers) # choose a walker at random
            if temp_order[walker_idx] != N_walkers-1: break # don't choose the highest temp walker
        walker_idx_adj = temp_order.index(temp_order[walker_idx]+1) # find the walker with the next highest temp
        if walkers[walker_idx].replica_exchange_move(walkers[walker_idx], \
                    walkers[walker_idx_adj]): # replica exchange move accepted, change temp_order
            print "Swapping T of walkers %i and %i" % (walker_idx,walker_idx_adj)
            temp_order_higher = temp_order[walker_idx_adj]
            temp_order[walker_idx_adj] = temp_order[walker_idx]
            temp_order[walker_idx] = temp_order_higher

# Example usage
if __name__ == "__main__":
    seq="RWWLYGPSLWVIFCMHYEKWGAGSTW"
    drive_simulation(7,0.10,0.40,1000,1000,protein=seq,potential="bt")
