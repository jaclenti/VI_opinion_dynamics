# The script has a simulator of a bounded confidence interval with backfire effect
# 

import numpy as np
from scipy.sparse import coo_array
from scipy.special import expit as sigmoid




class simulator_opinion_dynamics():
    
    def __init__(self, create_edges, opinion_update, num_parameters = 5, dim_edges = 4):
        self.create_edges = create_edges
        self.opinion_update = opinion_update
        self.num_parameters = num_parameters
        self.dim_edges = dim_edges
        
    def initialize_simulator(self, N, T, edge_per_t, X0 = [], seed = None):
        self.N = N
        self.T = T
        self.edge_per_t = edge_per_t

        if seed is not None:
            np.random.seed(seed) 
        
        if len(X0) == 0:
            self.X0 = np.random.random(N)
        else:
            self.X0 = X0
        
        
    def simulate_trajectory(self, parameters, start_edges = None, seed = None):
        assert len(parameters) == self.num_parameters, f"Required {self.num_parameters} parameters"
        if seed is not None:
            np.random.seed(seed) 
        
        if start_edges is None:
            start_edges = np.zeros([self.T-1, self.edge_per_t, self.dim_edges], dtype = np.int32)
        
        edges = np.zeros([self.T-1, self.edge_per_t, self.dim_edges], dtype = np.int32)
        u,v = start_edges[:,:,:2].transpose(2,1,0)

        X_t0 = self.X0.copy()
        X_list = [X_t0[None,:].copy()]
        
        # X_list = [X_t0.clip(1e-5, 1 - 1e-5)[None,:]]
        edges_list = []

        diff_X = X_t0[:,None] - X_t0[None,:]

        
        for t in range(self.T-1):
            edges_t = self.create_edges(self.N, self.edge_per_t, diff_X, 
                                        parameters, u = u[:,t], v = v[:,t])
            X_t1 = self.opinion_update(diff_X, X_t0.copy(), edges_t, self.N, parameters)
            
            diff_X = X_t1[:,None] - X_t1[None,:]
            edges_list.append(edges_t[None,:])
            X_t0 = np.clip(X_t1, 1e-5, 1 - 1e-5)
            
            # X_t1.clip(1e-5, 1 - 1e-5, out = X_t1)
            # X_t0 = X_t1
            # X_list.append(X_t1.clip(1e-5, 1 - 1e-5)[None,:])
            X_list.append(np.clip(X_t1, 1e-5, 1 - 1e-5)[None,:].copy())
            
            X_t0 = np.clip(X_t1, 1e-5, 1 - 1e-5)

        X = np.concatenate(X_list)
        edges = np.concatenate(edges_list)
        
        return X, edges

    

def create_edges_BC_backfire(N, edge_per_t, diff_X, parameters, u = np.zeros(1), v = np.zeros(1)):
    epsilon_plus, epsilon_minus, mu_plus, mu_minus, rho = parameters
    
    if (u.sum() + v.sum()) == 0:
        u, v = np.random.randint(low = 0, high = N, size = [2, edge_per_t], dtype = np.int32)
    # s_plus = np.int32(np.random.random(edge_per_t) < sigmoid(rho * (epsilon_plus - np.abs(diff_X[u,v]))))
    # s_minus = np.int32(np.random.random(edge_per_t) < sigmoid(-rho * (epsilon_minus - np.abs(diff_X[u,v]))))
    s_plus = (np.abs(diff_X[u,v]) < epsilon_plus) + 0.
    s_minus = (np.abs(diff_X[u,v]) > epsilon_minus) + 0.
    return np.concatenate([u[:,None], v[:,None], s_plus[:,None], s_minus[:,None]], axis = 1)
    
    
    
def opinion_update_BC_backfire(diff_X, X_t, edges_t, N, parameters):
    epsilon_plus, epsilon_minus, mu_plus, mu_minus, rho = parameters
    u, v, s_plus, s_minus = np.int32(edges_t).T
    s_plus, s_minus = np.float32(s_plus), np.float32(s_minus)
    
    diff_X_uv_plus = coo_array((diff_X[u, v] * s_plus, (u, v)), shape = (N, N))
    diff_X_uv_minus = coo_array((diff_X[u, v] * s_minus, (u, v)), shape = (N, N))
    
    updates_plus = mu_plus * (diff_X_uv_plus.sum(axis = 0))# - diff_X_uv_plus.sum(axis = 1))
    updates_minus = mu_minus * (diff_X_uv_minus.sum(axis = 0))# - diff_X_uv_minus.sum(axis = 1))
    # diff_X_uv_plus = diff_X[u, v] * s_plus
    # diff_X_uv_minus = diff_X[u, v] * s_minus
    
    # updates_plus = mu_plus * diff_X_uv_plus    
    # updates_minus = mu_minus * diff_X_uv_minus
    
    X_t += updates_plus
    X_t -= updates_minus

    # X_t = np.clip(X_t, 1e-5, 1-1e-5)
    
    return X_t

    
def simulate_trajectory(N, T, edge_per_t, epsilon_plus = None, epsilon_minus = None, mu_plus = None, mu_minus = None, rho = 32, seed = None):
    if seed is None:
        seed = np.random.randint(low = 0,high = 2 ** 20)

    if epsilon_plus is None:
        epsilon_plus = np.random.choice(5) * 0.1 + 0.05
        epsilon_minus = np.random.choice(5) * 0.1 + 0.05 + 0.5
        mu_plus = np.random.choice(5) * 0.02 + 0.01
        mu_minus = np.random.choice(5) * 0.02 + 0.01
    sim = simulator_opinion_dynamics(create_edges_BC_backfire, opinion_update_BC_backfire, num_parameters = 5, dim_edges = 4)
    sim.initialize_simulator(N, T, edge_per_t)
    return sim.simulate_trajectory((epsilon_plus, epsilon_minus, mu_plus, mu_minus, rho), seed = seed)

    