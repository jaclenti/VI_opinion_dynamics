# In this script we have the simulator of a Bounded Confidence model with backfire effect, with limited 
# influence of the feed (with HK dynamics). This means that at each step I sample a given number of agents u.
# For each u I sample a list of neighbors. Then, u select the first k of them, average the opinion of the selected
# neighbors and get closer/further such average opinion uncer BC dynamics with parameters epsilon_plus and epsilon_minus
# k represents the length of the feed influencing u.

import numpy as np
from tqdm import tqdm
from time import time
import jax.numpy as jnp
from scipy.special import expit as sigmoid

# max_f_possible is the total length of the feed
# obs_f (observed feed) is the number of users influencing u
# parameters = (epsilon_plus, epsilon_minus, mu_plus, mu_minus, rho)
# epsilon_plus = bounded confidence interval
# epsilon_minus = backfire effect threshold
# mu_plus = convergence rate (positive interactions)
# mu_minus = divergence rate (negative interactions)
# rho = steepness of the sigmoid

class simulator_opinion_dynamics():
    
    def __init__(self, create_edges, opinion_update, 
                 num_parameters = 5):
        self.create_edges = create_edges
        self.opinion_update = opinion_update
        self.num_parameters = num_parameters
        
    def initialize_simulator(self, N, T, edge_per_t,
                             X0 = [], seed = None):
        self.N = N
        self.T = T
        self.edge_per_t = edge_per_t
        if len(X0) == 0:
            self.X0 = np.random.random(N)
        else:
            self.X0 = X0
        
        
    def simulate_trajectory(self, parameters, max_f_possible, obs_f, start_edges = None, seed = None):
        assert len(parameters) == self.num_parameters, f"Required {self.num_parameters} parameters"
        if seed is not None:
            np.random.seed(seed) 
        if start_edges is None:
            start_edges = np.zeros([self.T-1, self.edge_per_t, max_f_possible + 3], dtype = np.int32)
        edges = np.zeros([self.T-1, self.edge_per_t, max_f_possible + 3], dtype = np.int32)
        u_max_obs = start_edges[:,:,:max_f_possible]
        v = start_edges[:,:,max_f_possible]
        
        X_t0 = self.X0
        X_list = [X_t0[None,:].copy()]
        edges_list = []
        
        for t in range(self.T-1):
            edges_t, diff_X = self.create_edges(X_t0, self.edge_per_t, parameters, max_f_possible, obs_f, u_max_obs[t], v[t])
            X_t1 = self.opinion_update(diff_X, X_t0.copy(), edges_t, parameters, max_f_possible, obs_f)

            edges_list.append(edges_t[None,:])
            X_list.append(X_t1[None,:])
            X_t0 = np.clip(X_t1, 1e-5, 1 - 1e-5)
        
        X = np.concatenate(X_list)
        edges = np.concatenate(edges_list)
        
        return X, edges

def create_edges_BC_higher_order(X_t, edge_per_t, parameters, max_f_possible, obs_f, u_max_obs, v):
    N = len(X_t)
    epsilon_plus, epsilon_minus, mu_plus, mu_minus, rho = parameters
    
    if u_max_obs.sum() == 0:
        v = np.random.randint(low = 0, high = N, size = [1, edge_per_t])
        u_max_obs = np.random.choice(N, size = [edge_per_t, max_f_possible], replace = True)
        u = u_max_obs[:,:obs_f]

    diff_X = (X_t[u].mean(axis = 1) - X_t[v])

    s_plus = (np.abs(diff_X) < epsilon_plus) + 0.
    s_minus = (np.abs(diff_X) > epsilon_minus) + 0.
    # s_plus = np.random.random(edge_per_t) < sigmoid(rho * (epsilon_plus - np.abs(diff_X)))
    # s_minus = np.random.random(edge_per_t) < sigmoid(-rho * (epsilon_minus - np.abs(diff_X)))
    
    max_edges_t = np.int32(np.concatenate([u_max_obs.T[None,:], v[:,None], s_plus[:,None], s_minus[:,None]], axis = 1)[0].T)

    return max_edges_t, diff_X[0,:]
        



def opinion_update_higher_order(diff_X, X_t, edges_t, parameters, max_f_possible, obs_f):
    epsilon_plus, epsilon_minus, mu_plus, mu_minus, rho = parameters
    N = len(X_t)

    v, s_plus, s_minus = edges_t[:,max_f_possible:].T
    u = edges_t[:,:obs_f].T
 
    s_plus, s_minus = np.float32(s_plus), np.float32(s_minus)
    
    diff_X_uv_plus = diff_X * s_plus
    diff_X_uv_minus = diff_X * s_minus
    
    updates_plus = mu_plus * diff_X_uv_plus
    X_t[v] += updates_plus
    
    updates_minus = mu_minus * diff_X_uv_minus
    X_t[v] -= updates_minus
    X_t = np.clip(X_t, 1e-5, 1-1e-5)
    
    return X_t


def simulate_trajectory(N, T, edge_per_t, max_f_possible, obs_f, epsilon_plus = None, epsilon_minus = None, mu_plus = 0.02, mu_minus = 0.02, rho = 32, seed = None):
    if epsilon_plus is None:
        epsilon_plus, epsilon_minus = np.random.random(2) / 2 + np.array([0., 0.5])
    sim = simulator_opinion_dynamics(create_edges_BC_higher_order, opinion_update_higher_order)
    sim.initialize_simulator(N, T, edge_per_t)
    X, edges = sim.simulate_trajectory((epsilon_plus, epsilon_minus, mu_plus, mu_minus, rho), max_f_possible, obs_f, seed = seed)

    return X, edges



def convert_edges_uvst(edges, with_jax = False):
    max_T, edge_per_t, num_s = edges.shape
    
    if with_jax:
        return jnp.concatenate((edges.reshape(((max_T) * edge_per_t, num_s)), 
                           jnp.array(jnp.repeat(jnp.arange(max_T), edge_per_t))[:, None]), axis = 1).T
    else:
        return np.concatenate((edges.reshape(((max_T) * edge_per_t, num_s)), 
                           np.array(np.repeat(np.arange(max_T), edge_per_t))[:, None]), axis = 1).T
