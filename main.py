from simulation import minimal_model as mm
from sampling import grid

import numpy as np

# # Grid testing
# eg = grid.EqualGrid()
# stack = eg.sample_stack([(0, 10), (0, 10)], 10)
# print(stack, len(stack))

# # Minimal model testing
alpha = np.log(5e-4/ 1e-4)/4.02359478109
B, D, g = 1.5, (alpha/2), 1.7
print(f"(init) B: {B:.3f}, D: {D:.10f}")
B_, D_ = mm.step(B,D,g, warm_up=0)