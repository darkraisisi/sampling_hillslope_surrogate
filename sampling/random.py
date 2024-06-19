from numpy import random, array, unravel_index

class Random:
    @staticmethod
    def sample_stack(features, n_points, random_state=0, **kwargs):
        random.seed(random_state)
        points_per_feature = []
        for (start, end) in features:
            assert end > start

            points_per_feature.append(random.uniform(start, end, n_points))
        return array(points_per_feature)

    @staticmethod
    def sample_stack_pdf(features, n_points, pdf, random_state=0, **kwargs):
        flat = pdf.flatten()
        sample_index = random.choice(a=flat.size, size=n_points, p=flat)
        adjusted_index = unravel_index(sample_index, pdf.shape, order='F')
        adjusted_index = array(list(zip(*adjusted_index)), dtype="float")
        
        for dim in range(2):
            min_bound, max_bound = features[dim]
            adjusted_index[:, dim] = min_bound + (max_bound - min_bound) * adjusted_index[:, dim] / pdf.shape[dim]
        return adjusted_index[:,0], adjusted_index[:,1]
