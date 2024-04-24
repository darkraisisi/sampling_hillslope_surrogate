from itertools import product

class EqualGrid:
    def __init__(self) -> None:
        pass

    def sample_stack(self, features, n_points):
        points_per_feature = []
        for (start, end) in features:
            assert end > start

            feature_width = end - start
            step_size = feature_width / n_points
            offset = step_size / 2

            points_per_feature.append([offset + (x*step_size) for x in range(n_points)])
        return list(product(*points_per_feature))
