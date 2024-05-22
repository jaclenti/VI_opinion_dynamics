# I implement a bounded confidence of opinion dynamics with backfire effect assuming that agents are divided in two categories, leaders and followers
# Each user has a role R \in \{L, F\}.
# Each user has an opinion Xu
# The parameters of the model are epsilon_plusL, epsilon_minusL, epsilon_plusF, epsilon_minusF, mu_plusL, mu_minusL, mu_plusF, mu_minusF
# At each time, I sample two users u and v
# A positive interaction occurs with probability P(sigma(rho * (epsilon_plusK - |Xu - Xv|))), where K = Rv
# A negative interaction occurs with probability P(1 - sigma(rho * (epsilon_minusK - |Xu - Xv|))), where K = Rv
# If there is a positive interaction Xv += mu_plusK * (Xu - Xv)
# If there is a negative interaction Xv -= clamp(mu_minusK * (Xu - Xv), [0,1])

from tqdm import tqdm

import jax.numpy as jnp
import numpy as np
import jax, jaxlib
from jax.scipy.special import expit as jax_sigmoid
from scipy.special import expit as sigmoid

class simulator_opinion_dynamics():
    
    def __init__(self, create_edges, opinion_update, 
                 num_parameters = 5, dim_edges = 4):
        self.create_edges = create_edges
        self.opinion_update = opinion_update
        self.num_parameters = num_parameters
        self.dim_edges = dim_edges
        
    def initialize_simulator(self, N, T, edge_per_t, initial_leaders = 0,
                             X0 = [], seed = None):
        if seed is None:
            seed = np.random.randint(low = 0,high = 2 ** 20)
        # key = random.PRNGKey(seed)

        self.N = N
        self.T = T
        self.edge_per_t = edge_per_t
        self.roles = np.random.permutation(np.hstack((np.ones(initial_leaders, dtype = np.int32), np.zeros(N - initial_leaders, dtype = np.int32))))
            # self.roles = random.permutation(key, np.hstack((np.ones(initial_leaders, dtype = np.int32), np.zeros(N - initial_leaders, dtype = np.int32))))
        
        if len(X0) == 0:
            self.X0 = np.random.random(N)
        else:
            self.X0 = X0
        
    def simulate_trajectory(self, parameters, roles = None, start_edges = None, seed = None):
        if roles is None:
            roles = self.roles
        assert len(parameters) == self.num_parameters, f"Required {self.num_parameters} parameters"
        if seed is None:
            seed = np.random.randint(low = 0,high = 2 ** 40)
        
        if start_edges is None:
            start_edges = np.zeros([self.T-1, self.edge_per_t, self.dim_edges], dtype = np.int32)
        edges = np.zeros([self.T-1, self.edge_per_t, self.dim_edges], dtype = np.int32)
        u,v = start_edges[:,:,:2].transpose(2,1,0)
        X_t0 = self.X0
        X_list = [X_t0[None,:].copy()]
        edges_list = []

        diff_X = X_t0[:,None] - X_t0[None,:]
        
        for t in range(self.T-1):
            
            edges_t = self.create_edges(roles, self.edge_per_t, diff_X, parameters, u = u[:,t], v = v[:,t], key = None)#subkeys[t])
            X_t1 = self.opinion_update(diff_X, X_t0.copy(), edges_t, roles, parameters)
            diff_X = X_t1[:,None] - X_t1[None,:]

            edges_list.append(edges_t[None,:])
            X_list.append(np.clip(X_t1, 1e-5, 1 - 1e-5)[None,:])
            X_t0 = np.clip(X_t1, 1e-5, 1 - 1e-5)

        X = np.concatenate(X_list)
        edges = np.concatenate(edges_list)
        
        return X, edges



def create_edges_BC_BF_roles(roles, edge_per_t, diff_X, parameters, u = np.zeros(1), v = np.zeros(1), key = None):
    # epsilon_plus = [epsilon_plus_F, epsilon_plus_L]
    # epsilon_minus = [epsilon_minus_F, epsilon_minus_L]
    # mu_plus = [mu_plus_F, mu_plus_L]
    # mu_minus = [mu_minus_F, mu_minus_L]
    if key is None:
        seed = np.random.randint(low = 0,high = 2 ** 40)
        # key = random.PRNGKey(seed)
        
    N = len(roles)
    epsilon_plus, epsilon_minus, mu_plus, mu_minus, rho = parameters
    if u.sum() == 0:
        u, v = np.random.randint(low = 0, high = N, size = [2, edge_per_t])
        # u, v = random.randint(key, minval = 0, maxval = N, shape = [2, edge_per_t])
    
    Rv = roles[v]
    
    # s_plus = np.random.random(edge_per_t) < sigmoid(rho * (epsilon_plus[Rv] - np.abs(diff_X[u,v])))
    # s_minus = np.random.random(edge_per_t) < sigmoid(-rho * (epsilon_minus[Rv] - np.abs(diff_X[u,v])))
    s_plus = (np.abs(diff_X[u,v]) < epsilon_plus[Rv]) + 0.
    s_minus = (np.abs(diff_X[u,v]) > epsilon_minus[Rv]) + 0.
    
    edges_t = np.concatenate([u[:,None], v[:,None], s_plus[:,None], s_minus[:,None]], axis = 1)
    return edges_t
        
def opinion_update_BC_BF_roles(diff_X, X_t, edges_t, roles, parameters):
    epsilon_plus, epsilon_minus, mu_plus, mu_minus, rho = parameters
    N = len(roles)

    u, v, s_plus, s_minus = np.int32(edges_t).T
    Rv = roles[v]
    s_plus, s_minus = np.float32(s_plus), np.float32(s_minus)
    diff_X_uv_plus = diff_X[u, v] * s_plus
    diff_X_uv_minus = diff_X[u, v] * s_minus
    
    updates_plus = mu_plus[Rv] * diff_X_uv_plus

    X_t[v] += updates_plus
    # X_t = X_t.at[v].set(X_t[v] + updates_plus)
    
    updates_minus = mu_minus[Rv] * diff_X_uv_minus
    X_t[v] -= updates_minus
    # X_t = X_t.at[v].set(X_t[v] - updates_minus)
    return X_t

def kappa_plus_from_epsilon(epsilon_plus, diff_X, rho, with_jax = False):
    if with_jax:
        return jax_sigmoid(rho * (epsilon_plus - jnp.abs(diff_X)))
    else:
        return sigmoid(rho * (epsilon_plus - np.abs(diff_X)))

def kappa_minus_from_epsilon(epsilon_minus, diff_X, rho, with_jax = False):
    if with_jax:
        return jax_sigmoid(-rho * (epsilon_minus - jnp.abs(diff_X)))
    else:
        return sigmoid(-rho * (epsilon_minus - np.abs(diff_X)))



def convert_edges_uvst(edges, with_jax = False):
    max_T, edge_per_t, num_s = edges.shape
    
    if with_jax:
        return jnp.concatenate((edges.reshape(((max_T) * edge_per_t, num_s)), jnp.array(jnp.repeat(jnp.arange(max_T), edge_per_t))[:, None]), axis = 1).T
    else:
        return np.concatenate((edges.reshape(((max_T) * edge_per_t, num_s)), np.array(np.repeat(np.arange(max_T), edge_per_t))[:, None]), axis = 1).T

def simulate_trajectory(N, T, edge_per_t, initial_leaders = 0, rho = 32, epsilon_plus = None, epsilon_minus = None, 
                        mu_plus = np.array([0.02,0.02]), mu_minus = np.array([0.02,0.02]), seed = None):
    if seed is None:
        seed = np.random.randint(low = 0,high = 2 ** 40)

    if epsilon_plus is None:
        epsilon_plus = np.random.random(2) / 2
        epsilon_minus = (np.random.random(2) + 1) / 2
    sim = simulator_opinion_dynamics(create_edges_BC_BF_roles, opinion_update_BC_BF_roles)
    sim.initialize_simulator(N, T, edge_per_t, initial_leaders, seed = seed)
    X, edges = sim.simulate_trajectory((epsilon_plus, epsilon_minus, mu_plus, mu_minus, rho))
        
    return X, edges, sim.roles
    































