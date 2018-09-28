# lattice_protein

TO DO:
-implement pull moves
-implement mutation moves
-implement order param calculations
-allow directional patches
-allow multiple proteins and add/remove proteins from box
-define length of equilibriation and sampling / SA phases
-there should be an even no. of walkers and only adjacent walkers allowed to move


BUGS
-chain crossings appear (rarely) - need to find what moves are causing this
-note that if no move type is available (e.g. no kink-jump move available and local_move_prob has kink-jump prob=1.) then the program
will be caught in an infinite loop
