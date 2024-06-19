from scipy.stats import qmc
import numpy as np

class LatinHyperCube:
    @staticmethod
    def sample_stack(features, n_points, random_state=0,  **kwargs):
        feature_scales = list(zip(*features))

        sampler = qmc.LatinHypercube(d=len(features), seed=random_state)
        sample = sampler.random(n=n_points)
        sample_scaled = qmc.scale(sample, *feature_scales)
        return sample_scaled[:,0], sample_scaled[:,1]
    
    @staticmethod
    def stratify_pdf(pdf, num_strata):
        # Calculate the cumulative distribution function (CDF) for each axis
        cdf_x = np.cumsum(np.sum(pdf, axis=1))
        cdf_y = np.cumsum(np.sum(pdf, axis=0))

        # Calculate the CDF split points for equal depth stratification
        split_points_x = np.linspace(0, 1, num_strata + 1)
        split_points_y = np.linspace(0, 1, num_strata + 1)

        # Find the indices in the CDF that correspond to these split points
        strata_indices_y = np.searchsorted(cdf_x, split_points_x[1:-1])
        strata_indices_x = np.searchsorted(cdf_y, split_points_y[1:-1])

        # Add the start and end indices to the strata indices
        strata_indices_y = np.concatenate(([0], strata_indices_y, [pdf.shape[0]]))
        strata_indices_x = np.concatenate(([0], strata_indices_x, [pdf.shape[1]]))
            
        return strata_indices_x, strata_indices_y
    
    
    @staticmethod
    def sample_stack_pdf(bounds, n_samples, pdf):
        """
        Perform Latin Hypercube Sampling with weighting based on a given PDF.

        Parameters:
        bounds (list of tuple): List of (min, max) bounds for each dimension.
        n_samples (int): Number of samples to generate.
        pdf (np.ndarray): PDF array corresponding to the grid.

        Returns:
        np.ndarray: Array of shape (n_samples, n_dimensions) with the sampled points.
        """
        n_dimensions = len(bounds)
        lhs_samples = np.zeros((n_samples, n_dimensions))
        num_strata = int(np.sqrt(n_samples))
        
        strata_indices_x, strata_indices_y = LatinHyperCube.stratify_pdf(pdf, num_strata)
        
        # Generate LHS samples
        sample_index = 0
        for i in range(num_strata):
            for j in range(num_strata):
                x_start, x_end = strata_indices_x[i], strata_indices_x[i + 1]
                y_start, y_end = strata_indices_y[j], strata_indices_y[j + 1]

                x_sample = np.random.uniform(x_start, x_end)
                y_sample = np.random.uniform(y_start, y_end)

                
                lhs_samples[sample_index, 0] = x_sample
                lhs_samples[sample_index, 1] = y_sample
                sample_index += 1
        
        # Shuffle within each dimension
        for dim in range(n_dimensions):
            np.random.shuffle(lhs_samples[:, dim])
        
        # Scale samples to the original bounds
        for dim in range(n_dimensions):
            min_bound, max_bound = bounds[dim]
            lhs_samples[:, dim] = min_bound + (max_bound - min_bound) * lhs_samples[:, dim] / pdf.shape[dim]

        return lhs_samples[:, 0], lhs_samples[:, 1]

class OrtogonalLatinHyperCube:
    @staticmethod
    def sample_stack(features, n_points, random_state=0, **kwargs):
        feature_scales = list(zip(*features))

        sampler = qmc.LatinHypercube(d=len(features), strength=2, seed=random_state)
        sample = sampler.random(n=n_points)
        sample_scaled = qmc.scale(sample, *feature_scales)
        return sample_scaled[:,0], sample_scaled[:,1]

