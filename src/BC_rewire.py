import numpy as np
from scipy.sparse import coo_array
from scipy.special import expit as sigmoid
import networkx as nx



class simulator_opinion_dynamics():
    
    def __init__(self, create_edges, opinion_update, link_rewire, num_parameters = 8, dim_edges = 6):
        self.create_edges = create_edges
        self.opinion_update = opinion_update
        self.num_parameters = num_parameters
        self.dim_edges = dim_edges
        self.link_rewire = link_rewire
        
    def initialize_simulator(self, N, T, edge_per_t, X0 = [], p = 0.1, seed = None):
        self.N = N
        self.T = T
        self.edge_per_t = edge_per_t
        
        connected_components = 2
        while connected_components > 1:
            self.G = nx.erdos_renyi_graph(n = N, p = p)
            connected_components = len(sorted(nx.connected_components(self.G)))
        # self.G = connect_graph(self.G, self.N)
        if seed is not None:
            np.random.seed(seed) 
        
        if len(X0) == 0:
            X0 = np.random.random(N)
        self.X0 = X0
        
        
    def simulate_trajectory(self, parameters, start_edges = None, seed = None, return_G = False):
        assert len(parameters) == self.num_parameters, f"Required {self.num_parameters} parameters"
        if seed is not None:
            np.random.seed(seed) 
        
        if start_edges is None:
            start_edges = np.zeros([self.T-1, self.edge_per_t, self.dim_edges], dtype = np.int32)
        
        edges = np.zeros([self.T-1, self.edge_per_t, self.dim_edges], dtype = np.int32)
        u,v = start_edges[:,:,:2].transpose(2,1,0)
        G_list = [self.G.copy()]
        X_t0 = self.X0.copy()
        X_list = [X_t0.clip(1e-5, 1 - 1e-5)[None,:].copy()]
        edges_list = []

        diff_X = X_t0[:,None] - X_t0[None,:]

        
        for t in range(self.T-1):
            edges_t = self.create_edges(self.N, self.edge_per_t, diff_X, self.G,
                                        parameters, u = u[:,t], v = v[:,t])
            X_t1 = self.opinion_update(diff_X, X_t0.copy(), edges_t, self.N, parameters)
            diff_X = X_t1[:,None] - X_t1[None,:]
            edges_list.append(edges_t[None,:])
            X_t1.clip(1e-5, 1 - 1e-5, out = X_t1)
            X_t0 = X_t1
            X_list.append(X_t1.clip(1e-5, 1 - 1e-5)[None,:])
            self.G = self.link_rewire(self.G, edges_t, self.N)
            G_list.append(self.G.copy())

        X = np.concatenate(X_list)
        edges = np.concatenate(edges_list)
        
        if return_G:
            return X, edges, G_list
        else:
            return X, edges

    
def create_edges_BC_backfire(N, edge_per_t, diff_X, G, parameters, u = np.zeros(1), v = np.zeros(1)):
    epsilon_plus, epsilon_minus, mu_plus, mu_minus, beta, q, rho_up, rho_lr = parameters
    
    if (u.sum() + v.sum()) == 0:
        u = np.random.randint(low = 0, high = N, size = [edge_per_t], dtype = np.int32)
        v = np.array([np.random.choice(G.adj[u_]) for u_ in u])
        # u, v = np.random.randint(low = 0, high = N, size = [2, edge_per_t], dtype = np.int32)
    # s_plus = np.int32(np.random.random(edge_per_t) < sigmoid(rho_up * (epsilon_plus - np.abs(diff_X[u,v]))))
    # s_minus = np.int32(np.random.random(edge_per_t) < sigmoid(-rho_up * (epsilon_minus - np.abs(diff_X[u,v]))))
    s_plus = (np.abs(diff_X[u,v]) < epsilon_plus) + 0
    s_minus = (np.abs(diff_X[u,v]) > epsilon_minus) + 0
    link_rewire = np.int32(np.random.random(edge_per_t) < sigmoid(-rho_lr * (beta - np.abs(diff_X[u,v]))))
    is_update = np.int32(np.random.random(edge_per_t) < q)
    return np.concatenate([u[:,None], v[:,None], s_plus[:,None], s_minus[:,None], link_rewire[:,None], is_update[:,None]], axis = 1)
    

def opinion_update_BC_backfire(diff_X, X_t, edges_t, N, parameters):
    epsilon_plus, epsilon_minus, mu_plus, mu_minus, beta, q, rho_up, rho_lr = parameters

    u, v, s_plus, s_minus, rewire, is_update = np.int32(edges_t).T
    s_plus, s_minus = np.float32(s_plus) * is_update, np.float32(s_minus) * is_update
    
    diff_X_uv_plus = coo_array((diff_X[u, v] * s_plus, (u, v)), shape = (N, N))
    diff_X_uv_minus = coo_array((diff_X[u, v] * s_minus, (u, v)), shape = (N, N))
    
    updates_plus = mu_plus * (diff_X_uv_plus.sum(axis = 0))
    updates_minus = mu_minus * (diff_X_uv_minus.sum(axis = 0))
    
    X_t += updates_plus
    X_t -= updates_minus

    X_t = np.clip(X_t, 1e-5, 1-1e-5)
    
    return X_t


def link_rewire_backfire(G, edges_t, N):  
    size = G.size()
    rewiring_edges = edges_t[(edges_t[:,-2] == 1)&(edges_t[:,-1] == 0)]
    if len(rewiring_edges) > 0:
        G.remove_edges_from(rewiring_edges[:,:2])
    
    G_new = G.copy()
    
    while G_new.size() != size:
        G_new = G.copy()
        swapping_edges = np.array([list(G_new.edges)[k] for k in np.random.choice(N, size = len(rewiring_edges), replace = False)])
        G_new.remove_edges_from(swapping_edges[:,:2])
        G_new.add_edges_from(np.vstack([rewiring_edges[:,0], swapping_edges[:,1]]).T)
        G_new.add_edges_from(np.vstack([swapping_edges[:,0], rewiring_edges[:,1]]).T)


    # create_new_edges = True
    # while create_new_edges:
    #     new_edges = np.array([rewiring_edges[:,0], np.random.randint(low = 0, high = N, size = len(rewiring_edges))]).T
    #     if (new_edges[:,0] == new_edges[:,1]).sum() == 0:
    #         create_new_edges = False    
    #         G.add_edges_from(new_edges)
    # G = connect_graph(G, N)
    return G_new

# def connect_graph(G, N):
#     G_degrees = np.array(nx.degree(G))
#     min_degree_node, min_degree_value = G_degrees[G_degrees[:,1].argsort()][0]
#     while min_degree_value == 0:
#         create_new_edges = True
#         while create_new_edges:
#             new_edges = np.array([min_degree_node, np.random.randint(low = 0, high = N)]).T
#             if (new_edges[0] != new_edges[1]):
#                 create_new_edges = False    
#                 G.add_edges_from(new_edges[None,:])
#                 G_degrees = np.array(nx.degree(G))
#                 min_degree_node, min_degree_value = G_degrees[G_degrees[:,1].argsort()][0]

#     return G


def simulate_trajectory(N, T, edge_per_t, epsilon_plus = None, epsilon_minus = None, beta = None, q = 0.5,
                        mu_plus = None, mu_minus = None, rho_up = 32, rho_lr = 4, seed = None, p_ER = 0.1):
    parameters = (epsilon_plus, epsilon_minus, mu_plus, mu_minus, beta, q, rho_up, rho_lr)

    if epsilon_plus is None:
        epsilon_plus = np.random.choice(5) * 0.1 + 0.05
        epsilon_minus = np.random.choice(5) * 0.1 + 0.05 + 0.5
        beta = np.random.choice(10) * 0.1 + 0.05
        mu_plus = np.random.choice(5) * 0.02 + 0.01
        mu_minus = np.random.choice(5) * 0.02 + 0.01
    sim = simulator_opinion_dynamics(create_edges_BC_backfire, opinion_update_BC_backfire, link_rewire_backfire)
    sim.initialize_simulator(N = N, T = T, edge_per_t = edge_per_t, p = p_ER)
    return sim.simulate_trajectory(parameters)

    
    
    
    
    
    
