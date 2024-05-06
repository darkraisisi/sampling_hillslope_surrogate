from numpy import meshgrid, linspace

class EqualStack:
    def __init__(self) -> None:
        pass

    def sample_stack(self, features, n_points, space_filling=False):
        points_per_feature = []
        for (start, end) in features:
            assert end > start
            
            points_per_feature.append(linspace(start, end, n_points))
        return meshgrid(*points_per_feature)

