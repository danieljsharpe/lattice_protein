import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams

res_colors = {
             # hydrophobic residues
             "F": "#e50000", "L": "#c20078", "W": "#7e1e9c", "V": "#f97306", "I": "#ffff14", "M": "#650021", \
             "Y": "#ff81c0", "A": "#ff028d", \
             # hydrophilic residues
             "N": "#15b01a", "T": "#029386", "S": "#01ff07", "Q": "#06470c", \
             # cationic residues
             "R": "#95d0fc", "K": "#00ffff", \
             # anionic residues
             "D": "#0343df", "E": "#0504aa", \
             # misc residues
             "C": "#653700", "H": "#001146", "P": "#000000", "G": "#929591"
             }

def plot_protein(protein, coords):
    rcParams['axes.labelsize'] = 14
    rcParams['axes.titlesize'] = 16
    rcParams['xtick.labelsize'] = 14
    rcParams['ytick.labelsize'] = 14
    rcParams['legend.fontsize'] = 14
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Computer Modern Roman']
    rcParams['text.usetex'] = True
    rcParams['grid.alpha'] = 0.0

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim3d(0, len(protein))
    ax.set_ylim3d(0, len(protein))
    ax.set_zlim3d(0, len(protein))
    ax.grid(True, which='both')
    plt.minorticks_on()
    ax.w_xaxis.gridlines.set_lw(1.0)
    ax.w_yaxis.gridlines.set_lw(1.0)
    ax.w_zaxis.gridlines.set_lw(1.0)
    ax.w_xaxis._axinfo.update({'grid' : {'color': (0.5019, 0.5019, 0.5019, 1.)}}) # colour in RGBA format
    ax.w_yaxis._axinfo.update({'grid' : {'color': (0.5019, 0.5019, 0.5019, 1.)}})
    ax.w_zaxis._axinfo.update({'grid' : {'color': (0.5019, 0.5019, 0.5019, 1.)}})
    ax.set_xticks(np.arange(0, len(protein)+1, 1))
    for i in range(len(protein)):
        ax.scatter(coords[i][0],coords[i][1],coords[i][2], c=res_colors[protein[i]], \
                   marker='o', s=180, edgecolors='face', zorder=2)
    ax.plot(np.array(coords)[:,0],np.array(coords)[:,1],np.array(coords)[:,2], linestyle='solid', \
            linewidth=2, color='black', marker=None, zorder=1)
    plt.show()

if __name__ == "__main__":
    coords = [[0,0,i] for i in range(4)]
    coords += [[0,1,3]]
    print coords
    plot_protein("FFFFF", coords)
