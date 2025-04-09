import numpy as np
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats import kendalltau
import time

# Growing Spheres
# https://github.com/thibaultlaugel/growingspheres/tree/master
# https://hal.sorbonne-universite.fr/hal-01905982/file/180115_final.pdf

class GSCFE:
    def __init__(self, obs_to_interprete, prediction_fn, target_class=None, caps=None, n_in_layer=2000, layer_shape='ball', first_radius=0.1, dicrease_radius=10, sparse=True, verbose=False):
        """
        Parameters:
            obs_to_interprete: The data point (as a vector) for which we want to generate a counterfactual explanation.

            prediction_fn: A function that accepts an input array, where rows are samples and columns are features, and returns an array of integers representing the predicted classes for rows from the input array.

            caps: A tuple specifying the minimum and maximum allowed values for each feature.

            target_class: The class label that we desire for the counterfactual instance. If set to None, the algorithm will simply search for any instance whose prediction differs from that of the original observation.

            n_in_layer: The number of candidate instances to generate at each iteration (or “layer”) of the search. A higher number increases the chances of finding a valid counterfactual but also increases computational cost.

            layer_shape: Defines the geometric configuration used for sampling the candidate counterfactuals.

                'ball': Samples points within a full hyperball (solid sphere), covering the entire volume around the observation.
                'ring': Samples points in a hollow spherical shell.
                'sphere': Samples points strictly on the surface of a hypersphere.
            
            first_radius: The initial radius of the hyperball used to sample candidate points around the observation, it determines how close the initial search is to the observation.

            dicrease_radius: A factor (greater than 1) that controls the rate at which the search radius is decreased (and then increased) during the exploration phase.

            sparse: A boolean flag indicating whether to perform feature selection after finding a candidate counterfactual. When True, the algorithm attempts to reduce the number of features changed from the original observation.

            verbose: Prints progress messages.
        """
        
        self.obs_to_interprete = obs_to_interprete
        self.prediction_fn = prediction_fn
        self.y_obs = prediction_fn(obs_to_interprete.reshape(1, -1))
        self.target_class = target_class
        self.caps = caps
        self.n_in_layer = n_in_layer

        if layer_shape in ['ball', 'ring', 'sphere']:
            self.layer_shape = layer_shape
        else:
            raise ValueError("Parameter layer_shape must be either 'ball', 'ring' or 'sphere'.")

        self.first_radius = first_radius

        if dicrease_radius > 1.0:
            self.dicrease_radius = dicrease_radius 
        else:
            raise ValueError("Parameter dicrease_radius must be > 1.0.")

        self.sparse = sparse
        
        self.verbose = verbose
        
        if int(self.y_obs[0]) != self.y_obs[0]:
            raise ValueError("Prediction function should return a class (integer).")

    def find_cfe(self):
        """
        Finds the decision border then performs projections to make the explanation sparse.
        """

        ennemies = self.explore_and_find_ennemies()
        
        if ennemies is None or len(ennemies) == 0:
            return (None, None, None)
        else:
            closest_ennemy = sorted(ennemies, key=lambda x: pairwise_distances(self.obs_to_interprete.reshape(1, -1), x.reshape(1, -1)))[0] 
            cfe = np.array(closest_ennemy)

            if self.sparse == True:
                cfe_sparse = self.do_feature_selection(closest_ennemy)
            else:
                cfe_sparse = cfe.copy()
            
            displacement_from_obs = cfe_sparse - self.obs_to_interprete

            print()
            print(f"dtypes: {type(cfe)}, {type(cfe_sparse)}, {type(displacement_from_obs)}")
            print(f"shapes: {cfe.shape}, {cfe_sparse.shape}, {displacement_from_obs.shape}")
            print(f"(min, max): ({cfe.min()}, {cfe.max()}), ({cfe_sparse.min()}, {cfe_sparse.max()}), ({displacement_from_obs.min()}, {displacement_from_obs.max()})")
            print()

            return (cfe, cfe_sparse, displacement_from_obs)
    
    def explore_and_find_ennemies(self):
        """
        Exploration of the feature space to find the decision boundary. Generation of instances in growing hyper layers.
        """

        n_ennemies = 999    # initial value to enter the loop
        radius = self.first_radius
        
        while n_ennemies > 0:
            first_layer = self.find_ennemies_in_layer(radius=radius, caps=self.caps, n=self.n_in_layer, first_layer=True)
            
            n_ennemies = first_layer.shape[0]

            if n_ennemies > 0:
                radius = radius / self.dicrease_radius

            if self.verbose == True:
                print(f"{n_ennemies} ennemies found in initial hyper{self.layer_shape}.")
            
                if n_ennemies > 0:
                    print("Zooming in...")
        else:
            if self.verbose == True:
                print(f"Expanding hyper{self.layer_shape}...")

            iteration = 1
            max_iterations = 10    # how many times we increase the size of our layer
            step = radius / self.dicrease_radius
            
            while n_ennemies <= 0 and iteration <= max_iterations:
                layer = self.find_ennemies_in_layer(radius=radius, step=step, caps=self.caps, n=self.n_in_layer, first_layer=False)       
                n_ennemies = layer.shape[0]
                radius = radius + step
                print(f"{n_ennemies} enemies found in iteration {iteration}.")
                iteration += 1
                
            if self.verbose == True:
                print(f"Final number of iterations: {iteration}.")

                if iteration > max_iterations:
                    print("Max iterations limit reached.")
        
        if self.verbose == True:
            print(f"Two last radii: {(radius - step, radius)}")
            print(f"Final number of ennemies: {n_ennemies}")
        
        time.sleep(3)
        
        return layer
    
    def find_ennemies_in_layer(self, radius=None, step=None, caps=None, n=2000, first_layer=False):
        """
        Basis for the algorithm: generates a hyper layer, labels it with the blackbox and returns the instances that are predicted to belong to the target class.
        """

        if first_layer:
            layer = GSCFE.generate_ball(self.obs_to_interprete, radius, n)
        else:
            if self.layer_shape == 'ball':
                layer = GSCFE.generate_ball(self.obs_to_interprete, radius + step, n)
            elif self.layer_shape == 'ring':
                segment = (radius, radius + step)
                layer = GSCFE.generate_ring(self.obs_to_interprete, segment, n)
            elif self.layer_shape == 'sphere':
                layer = GSCFE.generate_sphere(self.obs_to_interprete, radius + step, n)

        if caps != None:
            caps_fn = lambda x: min(max(x, caps[0]), caps[1])
            layer = np.vectorize(caps_fn)(layer)
            
        preds = self.prediction_fn(layer)
        
        if self.target_class == None:
            enemies_layer = layer[np.where(preds != self.y_obs)]
        else:
            enemies_layer = layer[np.where(preds == self.target_class)]
            
        return enemies_layer
    
    def do_feature_selection(self, cfe):
        """
        Projection step of the algorithm. Make projections to make (cfe - obs_to_interprete) sparse.
        Heuristic: sort the coordinates of np.abs(cfe - obs_to_interprete) in ascending order and project as long as it does not change the predicted class.
        """

        if self.verbose == True:
            print("Feature selection starts...")
        
        time.sleep(3)
            
        move_sorted = sorted(enumerate(abs(cfe - self.obs_to_interprete.flatten())), key=lambda x: x[1])
        move_sorted = [x[0] for x in move_sorted if x[1] > 0.0]
        
        out = cfe.copy()
        
        reduced = 0
        iteration = 1
        max_iterations = 1000
        
        for k in move_sorted:
            new_enn = out.copy()
            new_enn[k] = self.obs_to_interprete.flatten()[k]

            if self.target_class == None:
                condition_class = self.prediction_fn(new_enn.reshape(1, -1)) != self.y_obs
            else:
                condition_class = self.prediction_fn(new_enn.reshape(1, -1)) == self.target_class
                
            if condition_class:
                out[k] = new_enn[k]
                reduced += 1

            if iteration > max_iterations:
                break
            
            iteration += 1

            print(iteration)
                
        if self.verbose == True:
            print(f"Feature selection ended. Reduced {reduced} coordinates.")
        
        return out
    
    @staticmethod
    def get_distances(x1, x2):
        x1, x2 = x1.reshape(1, -1), x2.reshape(1, -1)
        euclidean = pairwise_distances(x1, x2)[0][0]
        same_coordinates = sum((x1 == x2)[0])
        #pearson = pearsonr(x1, x2)[0]
        kendall = kendalltau(x1, x2)
        out_dict = {
            'euclidean': euclidean,
            'sparsity': x1.shape[1] - same_coordinates,
            'kendall': kendall
        }
        return out_dict        

    @staticmethod
    def generate_ball(center, r, n):
        def norm(v):
            return np.linalg.norm(v, ord=2, axis=1)
        d = center.shape[1]
        u = np.random.normal(0, 1, (n, d + 2))    # an array of (d + 2) normally distributed random variables
        u = 1 / (norm(u)[:, None]) * u
        x = u[:, 0:d] * r    # take the first d coordinates
        x = x + center
        return x

    @staticmethod
    def generate_ring(center, segment, n):
        def norm(v):
            return np.linalg.norm(v, ord=2, axis=1)
        d = center.shape[1]
        z = np.random.normal(0, 1, (n, d))
        try:
            u = np.random.uniform(segment[0]**d, segment[1]**d, n)
        except OverflowError:
            raise OverflowError("Dimension too big for 'ring'. Please use 'ball' or 'sphere' instead.")
        r = u**(1 / float(d))
        z = np.array([a * b / c for a, b, c in zip(z, r, norm(z))])
        z = z + center
        return z

    @staticmethod
    def generate_sphere(center, r, n):    
        def norm(v):
            return np.linalg.norm(v, ord=2, axis=1)
        d = center.shape[1]
        z = np.random.normal(0, 1, (n, d))
        z = z / (norm(z)[:, None]) * r + center
        return z
