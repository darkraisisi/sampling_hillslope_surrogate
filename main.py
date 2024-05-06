from simulation import minimal_model as mm
from sampling import grid
from visualise import stream

import numpy as np

# Minimal model testing
g = 1.7
B_lim, D_lim = 2.9, 0.4

# Grid search
eg = grid.EqualStack()
D_grid, B_grid = eg.sample_stack([(0, D_lim), (0, B_lim)], 100)

# Run miminal model
(next_B, delta_B), (next_D, delta_D) = mm.step(B_grid, D_grid, g, warm_up=0)
# print((next_B[-1][-1], delta_B[-1][-1]), (next_D[-1][-1], delta_D[-1][-1]))

# Plot change
stream.show(D_grid, B_grid, delta_D, delta_B, g)