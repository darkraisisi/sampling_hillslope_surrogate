from numpy import meshgrid, random, array

class Random:
    def __init__(self) -> None:
        pass

    def sample_stack(self, features, n_points, space_filling=False, random_state=0):
        random.seed(random_state)
        points_per_feature = []
        for (start, end) in features:
            assert end > start

            points_per_feature.append(random.uniform(start, end, n_points))
        return array(points_per_feature)

