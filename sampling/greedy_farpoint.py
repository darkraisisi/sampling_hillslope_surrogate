import numpy as np
from sampling.grid import Grid

class GREEDYFP:
    @staticmethod
    def sample_stack(bounds, n_samples, previous_samples=None, scale=10, refresh_count=10, **kwargs):
        """
        Perform hybrid BC-GreedyFP sampling, including previous samples if provided.

        Parameters:
        bounds (list of tuple): List of (min, max) bounds for each dimension.
        n_samples (int): Number of new samples to select.
        previous_samples (array-like): Optional array of previous samples to consider.
        scale (int): Scale factor to determine the number of candidates for each sample.
        refresh_count (int): Number of samples after which to refresh the candidate set.

        Returns:
        tuple: Arrays representing the newly sampled points for each dimension.
        """
        n_candidates = n_samples * scale
        
        # Calculate scale for the second axis to match the range of the first axis.
        # Method is calculating distances, if axis not equal axis's dont get equally treated.
        first_range = bounds[0][1] - bounds[0][0]
        second_range = bounds[1][1] - bounds[1][0]
        axis_scale = first_range / second_range
        
        # Create equal bounds
        equal_bounds = [(0, first_range), (0, first_range)]
        
        # Scale points to equal bounds
        def scale_to_equal_bounds(points):
            scaled_points = points.copy()
            scaled_points[:, 1] *= axis_scale
            return scaled_points
        
        # Scale points back to original bounds
        def scale_from_equal_bounds(points):
            scaled_points = points.copy()
            scaled_points[:, 1] /= axis_scale
            return scaled_points

        # Initialize sampled points
        if previous_samples is not None:
            sampled_points = scale_to_equal_bounds(np.column_stack(previous_samples))
        else:
            sampled_points = np.empty((0, len(bounds)))

        def generate_candidates():
            return generate_candidates_random()

        def generate_candidates_random():
            return np.array([np.random.uniform(low, high, n_candidates) for (low, high) in equal_bounds]).T

        def generate_candidates_grid():
            return np.array(Grid.sample_grid(equal_bounds, n_candidates, False)).T.reshape(-1, len(bounds))

        def select_farthest(candidates, sampled_points):
            if sampled_points.shape[0] > 0:
                # Manhattan distance
                dists = np.sum(np.abs(candidates[:, np.newaxis] - sampled_points), axis=2)
                min_dists = np.min(dists, axis=1)
                weighted_dists = np.exp(min_dists)  # Exponential weighting
            else:
                weighted_dists = np.full(len(candidates), np.inf)
            max_weighted_dist = np.max(weighted_dists)
            max_idx = np.where(weighted_dists == max_weighted_dist)[0]  # Find indices of maximum distances
            farthest_idx = np.random.choice(max_idx)  # Randomly choose one of the indices multiple can be true.
            return candidates[farthest_idx]

        candidates = generate_candidates()
        new_sampled_points = np.empty((0, len(bounds)))
        for _ in range(n_samples):
            if sampled_points.shape[0] % refresh_count == 0:
                candidates = generate_candidates()
            farthest_point = select_farthest(candidates, sampled_points)
            sampled_points = np.vstack([sampled_points, farthest_point])
            new_sampled_points = np.vstack([new_sampled_points, farthest_point])
            candidates = np.delete(candidates, np.argmax(np.all(candidates == farthest_point, axis=1)), axis=0)
        
        return tuple(scale_from_equal_bounds(new_sampled_points).T)