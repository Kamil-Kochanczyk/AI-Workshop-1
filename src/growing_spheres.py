import numpy as np
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats import kendalltau
import time

# Growing Spheres
# https://github.com/thibaultlaugel/growingspheres/tree/master
# https://hal.sorbonne-universite.fr/hal-01905982/file/180115_final.pdf

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

def generate_ball(center, r, n):
    def norm(v):
        return np.linalg.norm(v, ord=2, axis=1)
    d = center.shape[1]
    u = np.random.normal(0, 1, (n, d + 2))  # an array of (d+2) normally distributed random variables
    norm_ = norm(u)
    u = 1 / norm_[:, None] * u
    x = u[:, 0:d] * r #take the first d coordinates
    x = x + center
    return x

def generate_ring(center, segment, n):
    def norm(v):
        return np.linalg.norm(v, ord=2, axis=1)
    d = center.shape[1]
    z = np.random.normal(0, 1, (n, d))
    try:
        u = np.random.uniform(segment[0]**d, segment[1]**d, n)
    except OverflowError:
        raise OverflowError("Dimension too big for hyperball sampling. Please use layer_shape='ball' (or 'sphere') instead.")
    r = u**(1 / float(d))
    z = np.array([a * b / c for a, b, c in zip(z, r, norm(z))])
    z = z + center
    return z

def generate_sphere(center, r, n):    
    def norm(v):
        return np.linalg.norm(v, ord=2, axis=1)
    d = center.shape[1]
    z = np.random.normal(0, 1, (n, d))
    z = z / (norm(z)[:, None]) * r + center
    return z

class GrowingSpheres:
    """
    obs_to_interprete: This is the data point (an array or vector) for which you want to generate a counterfactual explanation. Essentially, it is the instance whose prediction you are seeking to understand or challenge.

    prediction_fn:A function that accepts an input array and returns an integer representing its predicted class. This function is used to test whether generated counterfactual examples belong to the target class or not.

    caps: A tuple specifying the minimum and maximum allowed values for each feature. These limits are used to ensure that generated counterfactuals remain within a realistic or acceptable range (e.g., within the bounds of the training data).

    target_class: The class label that you desire for the counterfactual instance. If set to None, the algorithm will simply search for any instance whose prediction differs from that of the original observation. For binary classification, this is often the opposite class.

    n_in_layer: The number of candidate instances to generate at each iteration (or “layer”) of the search. A higher number increases the chances of finding a valid counterfactual but also increases computational cost.

    layer_shape: Defines the geometric configuration used for sampling the candidate counterfactuals.

        'ring': Samples points in a hollow spherical shell, which can help focus the search around the boundary of the decision region.
        'ball': Samples points within a full hyperball (solid sphere), covering the entire volume around the observation.
        'sphere': Samples points strictly on the surface of a hypersphere, emphasizing boundary exploration.
    
    first_radius: The initial radius of the hyperball used to sample candidate points around the observation. This radius determines how close the initial search is to the observation.

    dicrease_radius: A factor (greater than 1) that controls the rate at which the search radius is decreased (or adjusted) during the exploration phase. A larger value results in a quicker narrowing of the search space.

    sparse: A boolean flag indicating whether to perform feature selection after finding a candidate counterfactual. When True, the algorithm attempts to reduce the number of features changed from the original observation, leading to a more interpretable, sparse explanation.

    verbose: Prints progress messages.
    """

    def __init__(self,
                obs_to_interprete,
                prediction_fn,
                target_class=None,
                caps=None,
                n_in_layer=2000,
                layer_shape='ring',
                first_radius=0.1,
                dicrease_radius=10,
                sparse=True,
                verbose=False):

        self.obs_to_interprete = obs_to_interprete
        self.prediction_fn = prediction_fn
        self.y_obs = prediction_fn(obs_to_interprete.reshape(1, -1))
        self.target_class = target_class
        self.caps = caps
        self.n_in_layer = n_in_layer
        self.first_radius = first_radius

        if dicrease_radius <= 1.0:
            raise ValueError("Parameter dicrease_radius must be > 1.0")
        else:
            self.dicrease_radius = dicrease_radius 

        self.sparse = sparse

        if layer_shape in ['ball', 'ring', 'sphere']:
            self.layer_shape = layer_shape
        else:
            raise ValueError("Parameter layer_shape must be either 'ball', 'ring' or 'sphere'.")
        
        self.verbose = verbose
        
        if int(self.y_obs) != self.y_obs:
            raise ValueError("Prediction function should return a class (integer)")

    def find_counterfactual(self):
        """
        Finds the decision border then performs projections to make the explanation sparse.
        """

        ennemies_ = self.exploration()
        
        if ennemies_ is None or len(ennemies_) == 0:
            return None
        else:
            closest_ennemy_ = sorted(ennemies_, key=lambda x: pairwise_distances(self.obs_to_interprete.reshape(1, -1), x.reshape(1, -1)))[0] 
            self.e_star = closest_ennemy_

            if self.sparse == True:
                out = self.feature_selection(closest_ennemy_)
            else:
                out = closest_ennemy_
            return out
    
    def exploration(self):
        """
        Exploration of the feature space to find the decision boundary. Generation of instances in growing hyperspherical layers.
        """

        n_ennemies_ = 999
        radius_ = self.first_radius
        
        while n_ennemies_ > 0:
            first_layer_ = self.ennemies_in_layer(radius=radius_, caps=self.caps, n=self.n_in_layer, first_layer=True)
            
            n_ennemies_ = first_layer_.shape[0]
            radius_ = radius_ / self.dicrease_radius # radius gets dicreased no matter what, even if no enemy?

            if self.verbose == True:
                print(f"{n_ennemies_} ennemies found in initial hyperball.")
            
                if n_ennemies_ > 0:
                    print("Zooming in...")
        else:
            if self.verbose == True:
                print("Expanding hypersphere...")

            iteration = 0
            max_iterations = 10    # number of times we grow out layers
            step_ = radius_ / self.dicrease_radius
            
            while n_ennemies_ <= 0 and iteration <= max_iterations:
                print(f"{n_ennemies_}; {iteration}")
                layer = self.ennemies_in_layer(radius=radius_, step=step_, caps=self.caps, n=self.n_in_layer, first_layer=False)       
                n_ennemies_ = layer.shape[0]
                radius_ = radius_ + step_
                iteration += 1
                
            if self.verbose == True:
                if iteration > max_iterations:
                    print("Max iterations limit reached")
                print(f"Final number of iterations: {iteration}")
        
        if self.verbose == True:
            print("Final radius: ", (radius_ - step_, radius_))
            print("Final number of ennemies: ", n_ennemies_)
            time.sleep(5)
        
        return layer
    
    def ennemies_in_layer(self, radius=None, step=None, caps=None, n=1000, first_layer=False):
        """
        Basis for the algorithm: generates a hypersphere layer, labels it with the blackbox and returns the instances that are predicted to belong to the target class.
        """

        if first_layer:
            layer = generate_ball(self.obs_to_interprete, radius, n)
        else:
            if self.layer_shape == 'ball':
                layer = generate_ball(self.obs_to_interprete, radius + step, n)
            elif self.layer_shape == 'ring':
                segment = (radius, radius + step)
                layer = generate_ring(self.obs_to_interprete, segment, n)
            elif self.layer_shape == 'sphere':
                layer = generate_sphere(self.obs_to_interprete, radius + step, n)

        if caps != None:
            cap_fn_ = lambda x: min(max(x, caps[0]), caps[1])
            layer = np.vectorize(cap_fn_)(layer)
            
        preds_ = self.prediction_fn(layer)
        
        if self.target_class == None:
            enemies_layer = layer[np.where(preds_ != self.y_obs)]
        else:
            enemies_layer = layer[np.where(preds_ == self.target_class)]
            
        return enemies_layer
    
    def feature_selection(self, counterfactual):
        """
        Projection step of the algorithm. Make projections to make (e* - obs_to_interprete) sparse.
        Heuristic: sort the coordinates of np.abs(e* - obs_to_interprete) in ascending order and project as long as it does not change the predicted class.
        """

        if self.verbose == True:
            print("Feature selection...")
            time.sleep(5)
            
        move_sorted = sorted(enumerate(abs(counterfactual - self.obs_to_interprete.flatten())), key=lambda x: x[1])
        move_sorted = [x[0] for x in move_sorted if x[1] > 0.0]
        
        out = counterfactual.copy()
        
        reduced = 0
        iteration = 0
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
            
            iteration += 1
            if iteration > max_iterations:
                break

            print(iteration)
                
        if self.verbose == True:
            print(f"Reduced {reduced} coordinates")
            time.sleep(5)
        
        return out
    
    # def feature_selection_all(self, counterfactual):
    #     """
    #     Try all possible combinations of projections to make the explanation as sparse as possible. 
    #     Warning: really long!
    #     """

    #     if self.verbose == True:
    #         print("Grid search for projections...")
        
    #     for k in range(self.obs_to_interprete.size):
    #         print('==========', k, '==========')

    #         for combo in combinations(range(self.obs_to_interprete.size), k):
    #             out = counterfactual.copy()
    #             new_enn = out.copy()

    #             for v in combo:
    #                 new_enn[v] = self.obs_to_interprete[v]
                
    #             if self.prediction_fn(new_enn.reshape(1, -1)) == self.target_class:
    #                 print('bim')
    #                 out = new_enn.copy()
    #                 reduced = k
        
    #     if self.verbose == True:
    #         print(f"Reduced {reduced} coordinates")
        
    #     return out

class GSCFE:
    """
    Class for defining a Counterfactual Explanation.
    """

    def __init__(self, obs_to_interprete, prediction_fn, method='GS', target_class=None, random_state=None):
        self.obs_to_interprete = obs_to_interprete
        self.prediction_fn = prediction_fn
        self.method = method
        self.target_class = target_class
        self.random_state = check_random_state(random_state)
        self.methods_ = {'GS': GrowingSpheres}
        self.fitted = 0
        
    def fit(self, caps=None, n_in_layer=2000, layer_shape='ball', first_radius=0.1, dicrease_radius=10, sparse=True, verbose=False):
        """
        Find the counterfactual with the specified method.
        """
        
        cf = self.methods_[self.method](
            self.obs_to_interprete,
            self.prediction_fn,
            self.target_class,
            caps,
            n_in_layer,
            layer_shape,
            first_radius,
            dicrease_radius,
            sparse,
            verbose
        )
        self.enemy = cf.find_counterfactual()

        if self.enemy is None or len(self.enemy) == 0:
            self.e_star = None
            self.move = None
            self.fitted = 0
        else:
            self.e_star = cf.e_star
            self.move = self.enemy - self.obs_to_interprete
            self.fitted = 1
    
    # def distances(self):
    #     """
    #     Scores de distances entre l'obs et le counterfactual
    #     """

    #     if self.fitted < 1:
    #         raise AttributeError('CounterfactualExplanation has to be fitted first!')
        
    #     return get_distances(self.obs_to_interprete, self.enemy)
