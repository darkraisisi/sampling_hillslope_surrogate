from scipy.stats import qmc
from numpy import meshgrid
class LatinHyperCubeStack:
    def __init__(self) -> None:
        pass

    def sample_stack(self, features, n_points, **kwargs):
        print(features)
        feature_scales = list(zip(features[0]))

        sampler = qmc.LatinHypercube(d=len(features), **kwargs)
        sample = sampler.random(n=n_points)
        sample_scaled = qmc.scale(sample, *feature_scales)
        return meshgrid(sample_scaled[:,0], sample_scaled[:,1])
