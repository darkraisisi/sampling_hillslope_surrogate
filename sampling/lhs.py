from scipy.stats import qmc
from numpy import meshgrid

class LatinHyperCube:
    def __init__(self) -> None:
        pass

    def sample_stack(self, features, n_points, random_state=0,  **kwargs):
        print(features)
        feature_scales = list(zip(features[0]))

        sampler = qmc.LatinHypercube(d=len(features), seed=random_state, **kwargs)
        sample = sampler.random(n=n_points)
        sample_scaled = qmc.scale(sample, *feature_scales)
        return sample_scaled[:,0], sample_scaled[:,1]
    
class OrtogonalLatinHyperCube:
    def __init__(self) -> None:
        pass

    def sample_stack(self, features, n_points, random_state=0, **kwargs):
        print(features)
        feature_scales = list(zip(features[0]))

        sampler = qmc.LatinHypercube(d=len(features), strength=2, seed=random_state, **kwargs)
        sample = sampler.random(n=n_points)
        sample_scaled = qmc.scale(sample, *feature_scales)
        return sample_scaled[:,0], sample_scaled[:,1]

