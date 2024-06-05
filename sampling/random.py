from numpy import meshgrid, random, array

class Random:
    @staticmethod
    def sample_stack(features, n_points, random_state=0, **kwargs):
        random.seed(random_state)
        points_per_feature = []
        for (start, end) in features:
            assert end > start

            points_per_feature.append(random.uniform(start, end, n_points))
        return array(points_per_feature)

