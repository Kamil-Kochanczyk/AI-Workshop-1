import numpy as np
from sklearn.neighbors import KernelDensity

# FACE
# https://github.com/sharmapulkit/FACE-Feasible-Actionable-Counterfactual-Explanations
# https://arxiv.org/pdf/1909.09369

# ====================================
# Helper classes

class DistanceComputer:
    def __init__(self):
        pass

    def compute_distance(self, xi, xj):
        return np.linalg.norm(xi - xj, 2)

class PDFEstimator:
    EPSILON = 1e-8  # Small constant to avoid division by zero

    def __init__(self, distance_computer, bandwidth=0.5, kernel='tophat'):
        self.distance_computer = distance_computer
        self.kernel = KernelDensity(kernel=kernel, bandwidth=bandwidth)
        self.N = None
    
    def fit(self, X):
        self.N = len(X)
        # self.kernel.fit(X)
    
    def estimate_pdf(self, x, mode):
        # Kernel Density Estimator (KDE) or uniform distribution
        out = None
        if mode == "kde":
            log_density = self.kernel.score_samples([x])
            out = float(np.exp(log_density))
        elif mode == "uniform":
            out = 1 / self.N
        else:
            pass
        print(f"Estimated pdf value: {out}")
        return out
    
    def estimate_weight_cost_between(self, xi, xj, mode):
        mean = 0.5 * (xi + xj)
        distance = self.distance_computer.compute_distance(xi, xj)
        density_at_mean = self.estimate_pdf(mean, mode)
        out = (1 / (density_at_mean + PDFEstimator.EPSILON)) * distance
        print(f"Estimated weight cost between xi and xj: {out}")
        return out

class Constraints:
    def __init__(self, mutable=True, step_direction=0):
        self._mutable = mutable
        self._step_direction = step_direction

    @property
    def mutable(self):
        return self._mutable

    @mutable.setter
    def mutable(self, v):
        self._mutable = v

    @property
    def step_direction(self):
        return self._step_direction

    @step_direction.setter
    def step_direction(self, v):
        self._step_direction = v

class FeasibilitySet:
    def __init__(self):
        self._feasibility_set = {}

    def set_constraint(self, feat, mutability=True, step_direction=0):
        self._feasibility_set[feat] = Constraints(mutability, step_direction)

    def check_constraints(self, source, dest):
        if len(self._feasibility_set) == 0:
            return True

        delta = dest - source
        for feat in self._feasibility_set:
            if (delta[feat] != 0) and (self._feasibility_set[feat].mutable is False):
                return False
            if delta[feat] * self._feasibility_set[feat].step_direction < 0:
                return False

        return True

# End of helper classes
# ====================================

class FACECFE:
    """
    FACECFE class for generating counterfactual explanations (CFEs) using a graph-based approach.
    """

    MAX_VERTICES = 1000000

    def __init__(self, X_train, predict_fn, distance_computer, pdf_estimator, feasibility_set, epsilon=0.2, tp=0.5, td=0.001):
        """
        Initializes the FACECFE object.
        Parameters:
            X_train (list): The training dataset.
            predict_fn (callable): A function that predicts the label and gives predicted probabilities for a given input.
            distance_computer (object): An object with a method `compute_distance` to calculate distances between data points.
            pdf_estimator (object): An object with methods `fit` and `estimate_pdf` to estimate the probability density function.
            feasibility_set (object): An object with a method `check_constraints` to verify feasibility constraints between data points.
            epsilon (float): The maximum distance threshold for adding edges to the graph.
            tp (float): The minimum threshold for classifier predicted probability for choosing candidate CFEs.
            td (float): The minimum threshold for estimated probability density value for choosing cadidate CFEs.
        """

        self.X_train = X_train
        self.predict_fn = predict_fn
        self.distance_computer = distance_computer

        self.pdf_estimator = pdf_estimator
        self.pdf_estimator.fit(self.X_train)

        self.feasibility_set = feasibility_set

        self.epsilon = epsilon
        self.tp = tp
        self.td = td
        
        self.graph = None
        self.make_graph_adj_list()
        self.print_graph()
    
    def print_graph(self):
        print(f"\nGraph length: {len(self.graph)}")
        print("Graph:")
        for i in self.graph:
            print(f"{i}: {self.graph[i]}")
            print()
        input("Press any key to continue...\n")

    def make_graph_adj_list(self):
        """
        Constructs the adjacency list representation of the undirected, weighted graph based on the training data.
        """

        N = len(self.X_train)
        self.graph = {}

        for i in range(0, N):
            for j in range(i + 1, N):
                print(f"Considering edge wij, i = {i}, j = {j}")
                xi = self.X_train[i]
                xj = self.X_train[j]
                self.try_add_edge(xi, xj, i, j)
    
    def try_add_edge(self, xi, xj, i, j):
        """
        Attempts to add an edge between two vertices in the graph if they satisfy distance and feasibility constraints.
        Parameters:
            xi (np.ndarray): The first data point.
            xj (np.ndarray): The second data point.
            i (int): The index of the first data point.
            j (int): The index of the second data point.
        Returns:
            bool: True if the edge was added, False otherwise.
        """
        
        d = self.distance_computer.compute_distance(xi, xj)
        
        if ((d < self.epsilon) and (self.feasibility_set.check_constraints(xi, xj) is True)):
            self.add_edge(xi, xj, i, j, d)
            return True
        
        return False
    
    def add_edge(self, xi, xj, i, j, d):
        """
        Adds an edge between two vertices in the graph with a weight based on distance and cost.
        Parameters:
            xi (np.ndarray): The first data point.
            xj (np.ndarray): The second data point.
            i (int): The index of the first data point.
            j (int): The index of the second data point.
            d (float): The distance between the two data points.
        """

        wij = d * self.pdf_estimator.estimate_weight_cost_between(xi.flatten(), xj.flatten(), "uniform")
        
        if (i in self.graph):
            self.graph[i][j] = wij # Add edge to the graph
        else:
            self.graph[i] = {j: wij}
        
        if (j in self.graph):
            self.graph[j][i] = wij # Add edge to the graph
        else:
            self.graph[j] = {i: wij}

    def find_shortest_path(self, source, target):
        """
        Finds the shortest path between two vertices in the graph using a greedy approach.
        Parameters:
            source (int): The source vertex ID.
            target (int): The target vertex ID.
        Returns:
            tuple: A tuple containing the shortest path as a list of vertex IDs and the total path cost.
                   If no path exists, returns ([], -1).
        """

        path = [source]
        path_cost = 0
        current = source
        visited = []

        while (current is not target):
            visited.append(current)
            if not (current in self.graph):
                return [], -1

            neighbour_connections = self.graph[current]
            minimum_cost = float("inf")
            closest = -1

            for n in neighbour_connections:
                if ((neighbour_connections[n] < minimum_cost) and not (n in visited)):
                    minimum_cost = neighbour_connections[n]
                    closest = n

            if (closest == -1):
                return [], -1

            path.append(closest)
            path_cost += minimum_cost
            current = closest

        return path, path_cost

    def get_cfe_candidates(self, target_label):
        """
        Identifies counterfactual explanation (CFE) candidates based on the target label.
        Parameters:
            target_label (int): The target label for which CFEs are sought.
        Returns:
            dict: A dictionary where keys are candidate vertex IDs and values are the corresponding data points.
        """

        candidates = {}

        for i in range(len(self.X_train)):
            np_img_row = self.X_train[i]
            label, probabilities = self.predict_fn(np_img_row)
            target_label_predicted_probability = probabilities[int(target_label)]

            if (label == target_label and 
                target_label_predicted_probability > self.tp and 
                self.pdf_estimator.estimate_pdf(np_img_row.flatten(), "uniform") > self.td):
                candidates[i] = np_img_row
        
        return candidates

    def get_cfe(self, source):
        """
        Generates a counterfactual explanation (CFE) for a given source data point.
        Parameters:
            source (np.ndarray): The source data point for which a CFE is to be generated.
        Returns:
            tuple: A tuple containing the CFE data point, the path cost, and the path as a list of vertex IDs.
                   If no CFE is found, returns (-1, float('inf'), None).
        """

        assert (self.graph is not None)

        source_vertex_id = np.random.randint(0, FACECFE.MAX_VERTICES)
        while (source_vertex_id in self.graph):
            source_vertex_id = np.random.randint(0, FACECFE.MAX_VERTICES)

        for i in range(len(self.X_train)):
            xi = self.X_train[i]
            self.add_edge(xi, source, i, source_vertex_id, self.distance_computer.compute_distance(xi, source))
        
        print(f"\nNew vertex added with index {source_vertex_id}")
        print(f"New vertex: {self.graph[source_vertex_id]}")
        input("\nPress any key to continue...\n")
        
        target_label = not self.predict_fn(source)[0]
        cfe_candidates = self.get_cfe_candidates(target_label)
        
        cfe_path_cost = float("inf")
        cfe = -1
        cfe_path = None

        for candidate_id in cfe_candidates:
            candidate = cfe_candidates[candidate_id]
            path, path_cost = self.find_shortest_path(source_vertex_id, candidate_id)

            if (path_cost == -1):
                continue
            
            if (path_cost < cfe_path_cost):
                cfe = candidate
                cfe_path_cost = path_cost
                cfe_path = path
        
        return cfe, cfe_path_cost, cfe_path
