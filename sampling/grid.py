from numpy import meshgrid, linspace, sqrt

class Grid:
    def __init__(self) -> None:
        pass

    def sample_stack(self, features, n_points, edge_bound=True, **kwargs):
        n_points = int(sqrt(n_points))

        points_per_feature = []
        if edge_bound:
            for (start, end) in features:
                assert end > start

                points_per_feature.append(linspace(start, end, n_points))
        else:
            for (start, end) in features:
                assert end > start
                offset = (end - start) / (n_points - 1) / 2
                points_per_feature.append(linspace(start + offset, end - offset, n_points))

        return [x.flatten() for x in meshgrid(*points_per_feature)]

