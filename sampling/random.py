import numpy as np

class Random:
    @staticmethod
    def sample_stack(features, n_points, random_state=0, **kwargs):
        # random.seed(random_state)
        points_per_feature = []
        for (start, end) in features:
            assert end > start

            points_per_feature.append(np.random.uniform(start, end, n_points))
        return np.array(points_per_feature)
    
    @staticmethod
    def scale_min_max(x):
        min_val = np.min(x)
        max_val = np.max(x)
        return (x - min_val) / (max_val - min_val)
    
    @staticmethod
    def sample_stack_pdf(features, n_points, pdf, random_state=0, beta=2, **kwargs):
        flat = pdf.flatten()
        if beta > 0:
            flat = np.power(flat, beta)
            flat = Random.scale_min_max(flat)
            flat = flat / sum(flat)

        sample_index = np.random.choice(a=flat.size, size=n_points, p=flat)
        adjusted_index = np.unravel_index(sample_index, pdf.shape, order='F')
        adjusted_index = np.array(list(zip(*adjusted_index)), dtype="float")
        
        for dim in range(2):
            min_bound, max_bound = features[dim]
            adjusted_index[:, dim] = min_bound + (max_bound - min_bound) * adjusted_index[:, dim] / pdf.shape[dim]
        return adjusted_index[:,0], adjusted_index[:,1]
