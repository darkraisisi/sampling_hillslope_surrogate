from simulation import minimal_model
from sampling import grid

eg = grid.EqualGrid()
stack = eg.sample_stack([(0, 10), (0, 10)], 10)
print(stack, len(stack))