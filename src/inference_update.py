import sys 
sys.path += ["../src"]
import BC_leaders, BC_update
import numpy as np
from tqdm import tqdm
from time import time
import pickle
from glob import glob
from pyABC_ import pyabc
from scipy.special import expit as np_sigmoid

import os
from tempfile import gettempdir
from pyABC_.pyabc.sampler import SingleCoreSampler
from jax.scipy.special import expit as sigmoid
import jax
import jax.numpy as jnp
from jax.experimental import sparse
from numpyro.infer import SVI, Trace_ELBO, TraceGraph_ELBO, MCMC, NUTS
from numpyro.infer.autoguide import AutoNormal, AutoBNAFNormal, AutoIAFNormal
from numpyro import distributions
import numpyro
from numpyro.optim import Adam
import jax.random as random
from datetime import timedelta
# numpyro.set_platform("gpu")
from diptest import dipstat
from scipy.stats import kurtosis, skew

######## numpyro #############
def count_s_from_edge(e): # e = (u,v,s,t)
    e = jnp.int32(e)
    e_unique_pairs, e_unique_count = jnp.unique(e, axis = 0, return_counts = True)
    e_unique_weigths = e_unique_pairs[:,2] * e_unique_count
    return e_unique_weigths, e_unique_pairs[:,:2]

def edges_coo_mu(edges, N):
    M_list = [sparse.BCOO(count_s_from_edge(edges[t]),
                          shape = jnp.array([N,N])).todense() for t in range(len(edges))]
    return M_list

def compute_Xt(Xt, M_plus_t_dense, M_minus_t_dense, mu_plus, mu_minus):
    # M_plus_t_dense, M_minus_t_dense  = M_plus_t.todense(), M_minus_t.todense()
    diff_X_plus = (Xt * M_plus_t_dense.T).sum(axis = 1) - (Xt * M_plus_t_dense).sum(axis = 0)
    diff_X_minus = (Xt * M_minus_t_dense.T).sum(axis = 1) - (Xt * M_minus_t_dense).sum(axis = 0)
    updates_plus = mu_plus * diff_X_plus
    updates_minus = mu_minus * diff_X_minus
    Xt = Xt + updates_plus - updates_minus
    Xt_data_clipped = jnp.clip(Xt.copy(), 0, 1)
    Xt = Xt_data_clipped
    return Xt


def update_X(M_plus, M_minus, mu_plus, mu_minus, Xt, X_list = [], t = 0, T_max = 0):
    M_plus_t, M_minus_t = M_plus[t], M_minus[t]
    Xt = compute_Xt(Xt, M_plus_t, M_minus_t, mu_plus, mu_minus)
    X_list.append(Xt[None,:])
    t += 1
    if t < T_max-1:
        update_X(M_plus, M_minus, mu_plus, mu_minus, Xt, X_list, t, T_max)
        
    return X_list


########### 
# def compute_X_from_X0_params(X0, edges_iter, epsilon_plus, epsilon_minus, mu_plus, mu_minus, rho):
#     X_list = []
#     Xt = X0.copy()
#     N = len(Xt)
    
#     while True:
#         edges_t = next(edges_iter, None)
#         if edges_t is None:
#             break
        
#         u,v,_,_ = edges_t.T
#         u,v = u.astype(int),v.astype(int)
#         diff_X = Xt[u] - Xt[v]
#         # s_plus = ((np.random.rand(edge_per_t) < np_sigmoid(rho * (epsilon_plus - np.abs(diff_X))))) + 0
#         # s_minus = ((np.random.rand(edge_per_t) < np_sigmoid(-rho * (epsilon_minus - np.abs(diff_X))))) + 0
#         s_plus =  (np.abs(diff_X) > epsilon_plus) + 0
#         s_minus = (np.abs(diff_X) < epsilon_minus) + 0

#         updates_plus = mu_plus * s_plus * diff_X
#         updates_minus = mu_minus * s_minus * diff_X
#         Xt = Xt.at[v].set(Xt[v] + updates_plus - updates_minus)
#         Xt[v] = np.clip(Xt[v], 1e-5, 1 - 1e-5)
        
#         X_list.append(Xt)

#     return np.concatenate(X_list)

# def initialize_training(X, edges, rho = 32):
#     T, N = X.shape    
#     edges_iter = (edges_t for edges_t in edges)
#     u,v,s_plus,s_minus,t = BC_leaders.convert_edges_uvst(edges)
#     s_plus, s_minus = jnp.float32(s_plus), jnp.float32(s_minus)

#     X0 = jnp.array(X[0])
#     return {"u": u, "v": v, "s_plus": s_plus, "s_minus": s_minus, "t": t,
#             "N": N, "T": T, "rho": rho, "X0": X0, 
#             "edges_iter": edges_iter}
#############
def initialize_training(X, edges, rho = 32):
    T, N = X.shape    
    u,v,s_plus,s_minus,t = BC_leaders.convert_edges_uvst(edges)
    s_plus, s_minus = jnp.float32(s_plus), jnp.float32(s_minus)

    M_plus_list = edges_coo_mu(edges[:,:,[0,1,2]], N)
    M_minus_list = edges_coo_mu(edges[:,:,[0,1,3]], N)

    X0 = jnp.array(X[0])
    return {"u": u, "v": v, "s_plus": s_plus, "s_minus": s_minus, "t": t,
            "N": N, "T": T, "rho": rho, "X0": X0, 
            "M_plus_list": M_plus_list, "M_minus_list": M_minus_list}

def model(data):
    X0,u,v,s_plus, s_minus,t, M_plus_list, M_minus_list, rho, N, T = [data[k] for k in ["X0","u","v","s_plus", "s_minus","t",
                                                                            "M_plus_list", "M_minus_list", "rho", "N", "T"]]
    # X0,u,v,s_plus, s_minus,t, edges_iter, rho, N, T = [data[k] for k in ["X0","u","v","s_plus", "s_minus","t",
    #                                                                         "edges_iter", "rho", "N", "T"]]
    
    dim = 4
    dist = distributions.Normal(jnp.zeros(dim), jnp.ones(dim)).to_event(1)
    params = numpyro.sample("theta", dist)
    
    theta = params[:4]
    epsilon_plus, epsilon_minus, mu_plus, mu_minus = sigmoid(theta) /  jnp.array([2,2,10,10]) + jnp.array([0.,.5, 0., 0.])

    ############
    X_sparse_list = update_X(M_plus_list, M_minus_list, mu_plus, mu_minus, X0, 
                             X_list = [X0[None,:]], t = 0, T_max = T)
    X = jnp.concatenate(X_sparse_list)
    ###########
    # X = compute_X_from_X0_params(X0, edges_iter, epsilon_plus, epsilon_minus, mu_plus, mu_minus, rho)
    #########
    
    u,v,t = u.astype(int),v.astype(int),t.astype(int)
    diff_X = X[t,u] - X[t,v]

    kappas_plus = BC_leaders.kappa_plus_from_epsilon(epsilon_plus, diff_X, rho, with_jax = True)
    kappas_minus = BC_leaders.kappa_minus_from_epsilon(epsilon_minus, diff_X, rho, with_jax = True)
    kappas_ = jnp.concatenate([kappas_minus, kappas_plus])
    s = jnp.concatenate([s_minus, s_plus])

    with numpyro.plate("data", s.shape[0]):
        numpyro.sample("obs", distributions.Bernoulli(probs = kappas_), obs = s)

def train_mcmc(X, edges, intermediate_samples = None, rho = 32, num_chains = 1,
               warmup_samples = None, n_samples = 400, progress_bar = False, id = None, timeout = 3600):
    if intermediate_samples is None:
        intermediate_samples = n_samples
    if warmup_samples is None:
        warmup_samples = intermediate_samples

    data = initialize_training(jnp.array(X), jnp.array(edges), rho = rho)
    key = random.PRNGKey(0)
    mcmc = MCMC(NUTS(model), num_warmup = warmup_samples, num_chains = num_chains, 
                num_samples = intermediate_samples, progress_bar = progress_bar)
    res = []
    tot_time = 0
    for _ in range(int(n_samples / intermediate_samples)):
        t0 = time()
        mcmc.run(key, data)
        t1 = time()
        tot_time += t1 - t0

        mcmc.post_warmup_state = mcmc.last_state
        key = mcmc.post_warmup_state.rng_key
        
        mcmc_samples = mcmc.get_samples()
        param_mean, param_std = analyse_samples(mcmc_samples["theta"])
        res.append({"param_mean": param_mean,
                    "param_std": param_std,
                    "tot_time": tot_time,
                    "n_simulations": None,
                    "method": "mcmc",
                    "n_steps": None,
                    "n_samples": intermediate_samples * (_ + 1),
                    "id": id})
        if tot_time > timeout:
            break

    return res

def train_svi(X, edges, guide_family = "normal", rho = 32,
              n_steps = 4000, intermediate_steps = None, lr = 0.01, 
              progress_bar = False, id = None, timeout = 3600):
    if intermediate_steps is None:
        intermediate_steps = n_steps
    
    if guide_family == "normal":
        guide = AutoNormal(model)
    if guide_family == "NF":
        guide = AutoBNAFNormal(model, num_flows = 1)
        n_steps = int(n_steps / 2)
        intermediate_steps = int(intermediate_steps / 2)
    
    data = initialize_training(jnp.array(X), jnp.array(edges), rho = rho)
    optimizer = Adam(step_size = lr)
    svi = SVI(model, guide, optimizer, loss = TraceGraph_ELBO())
    res = []
    last_state = None

    tot_time = 0
    
    for _ in range(int(n_steps / intermediate_steps)):
        t0 = time()
        svi_results = svi.run(random.PRNGKey(0), intermediate_steps, data, init_state = last_state, progress_bar = progress_bar)
        t1 = time()
        tot_time += t1 - t0

        theta_samples = guide.sample_posterior(random.PRNGKey(0), svi_results.params, sample_shape = (200,))["theta"]
        param_mean, param_std = analyse_samples(theta_samples)
        
        res_analysis = {"param_mean": param_mean,
                        "param_std": param_std,
                        "tot_time": tot_time,
                        "n_simulations": None,
                        "method": "svi" + guide_family,
                        "n_steps": intermediate_steps * (_ + 1),
                        "n_samples": None,
                        "id": id
                        }
        res.append(res_analysis)

        last_state = svi_results.state
        if tot_time > timeout:
            break

    return res


######## pyabc #############
# def create_s_update_X(X_t, edges_iter, edge_per_t, parameters, rho,
#                      summary_statistics_list, X_list):
#     edges_t = next(edges_iter, None)
#     if edges_t is not None:
#         epsilon_plus,epsilon_minus, mu_plus, mu_minus = epsilons_from_theta(parameters, dict_theta = True, numpy = True)
#         u,v,_,_ = edges_t.T

#         diff_X = X_t[u] - X_t[v]
#         s_plus = ((np.random.rand(edge_per_t) < np_sigmoid(rho * (epsilon_plus - np.abs(diff_X))))) + 0
#         s_minus = ((np.random.rand(edge_per_t) < np_sigmoid(-rho * (epsilon_minus - np.abs(diff_X))))) + 0

#         updates_plus = mu_plus * s_plus * diff_X 
#         updates_minus = mu_minus * s_minus * diff_X 
#         X_t[v] += updates_plus - updates_minus
#         X_t[v] = np.clip(X_t[v], 1e-5, 1 - 1e-5)
#         X_list.append(X_t[None,:].copy())
#         summary_statistics_list.append(np.concatenate([u[None,:],v[None,:],s_plus[None,:], s_minus[None,:]])[None,:])
#         create_s_update_X(X_t, edges_iter, edge_per_t, parameters, rho, summary_statistics_list, X_list)
#     edges_sim = np.concatenate(summary_statistics_list).transpose(0,2,1)
#     return {"s_plus_sum": edges_sim[:,:,-2].sum(axis = 1), 
#             "s_minus_sum": edges_sim[:,:,-1].sum(axis = 1)}

def create_summary_statistics(X0, edges_iter, edge_per_t, parameters, rho):
    summary_statistics_list = []
    Xt = X0.copy()
    N = len(Xt)
    
    while True:
        edges_t = next(edges_iter, None)
        if edges_t is None:
            break
        epsilon_plus,epsilon_minus, mu_plus, mu_minus = epsilons_from_theta(parameters, dict_theta = True, numpy = True)
        u,v,_,_ = edges_t.T
        u,v = u.astype(int),v.astype(int)
        diff_X = Xt[u] - Xt[v]
        # s_plus = ((np.random.rand(edge_per_t) < np_sigmoid(rho * (epsilon_plus - np.abs(diff_X))))) + 0
        # s_minus = ((np.random.rand(edge_per_t) < np_sigmoid(-rho * (epsilon_minus - np.abs(diff_X))))) + 0
        s_plus =  (np.abs(diff_X) < epsilon_plus) + 0
        s_minus = (np.abs(diff_X) > epsilon_minus) + 0

        updates_plus = mu_plus * s_plus * diff_X 
        updates_minus = mu_minus * s_minus * diff_X 
        Xt[v] += updates_plus - updates_minus
        Xt[v] = np.clip(Xt[v], 1e-5, 1 - 1e-5)
            
        summary_statistics_list.append(np.concatenate([u[None,:],v[None,:],s_plus[None,:], s_minus[None,:]])[None,:])

    edges_sim = np.concatenate(summary_statistics_list).transpose(0,2,1)
    return {"s_plus_sum": edges_sim[:,:,-2].sum(axis = 1), 
            "s_minus_sum": edges_sim[:,:,-1].sum(axis = 1)}

    

def create_trajectory(X0, edges, parameters, rho):
    X0 = X0.copy()
    edges_iter = (edges_t for edges_t in edges)
    T, edge_per_t, _ = edges.shape
    summary_statistics = create_summary_statistics(X0, edges_iter, edge_per_t, parameters, rho)
    # summary_statistics = create_s_update_X(X0, edges_iter, edge_per_t, parameters, rho, [], [X0[None,:].copy()])
    return summary_statistics

def sim_trajectory_X0_edges(X0, edges, rho):
    return lambda parameters: create_trajectory(X0, edges, parameters, rho)



def train_abc(X, edges, populations_budget = 10, intermediate_populations = None,
              population_size = 200, rho = 32, id = None, timeout = 3600):
    if intermediate_populations is None:
        intermediate_populations = populations_budget
    
    T = len(X)
    res = []
    tot_time = 0
    model_abc = sim_trajectory_X0_edges(X[0], edges, rho)
    prior = pyabc.Distribution(
                theta0=pyabc.RV("norm", 0, 1),
                theta1=pyabc.RV("norm", 0, 1),
                theta2=pyabc.RV("norm", 0, 1),
                theta3=pyabc.RV("norm", 0, 1))
    distance = pyabc.PNormDistance(2)
    obs = {"s_plus_sum": edges[:,:,-2].sum(axis = 1), 
           "s_minus_sum": edges[:,:,-1].sum(axis = 1)}
    abc = pyabc.ABCSMC(model_abc, prior, distance, population_size = population_size)#, sampler = SingleCoreSampler())
    db = "sqlite:///" + os.path.join(gettempdir(), f"{id}_update_test.db")
    history = abc.new(db, obs)
    run_id = history.id
    for _ in range(int(populations_budget / intermediate_populations)):
        abc_continued = pyabc.ABCSMC(model_abc, prior, distance, population_size = population_size)#, sampler = SingleCoreSampler())
        abc_continued.load(db, run_id)
        t0 = time()
        history = abc_continued.run(max_nr_populations = intermediate_populations,
                                    minimum_epsilon = 5 * (T ** (1/2)),
                                    max_walltime = timedelta(hours = 3))
        t1 = time()
        tot_time += (t1 - t0)
        theta_samples = jnp.array(history.get_distribution()[0])

        param_mean, param_std = analyse_samples(theta_samples)
        res_analysis = {"param_mean": param_mean,
                        "param_std": param_std,
                        "tot_time": tot_time,
                        "n_simulations": history.total_nr_simulations,
                        "method": "abc",
                        "n_steps": None,
                        "n_samples": None,
                        "id": id
                        }
        res.append(res_analysis)
        if tot_time > timeout:
            break
    return res



def count_interactions(X, edges):
    T, N = X.shape
    _,edge_per_t,_ = edges.shape
    
    pos_interactions_plus, pos_interactions_minus = edges[:,:,2].sum(), edges[:,:,3].sum()
    tot_interactions = (T - 1) * edge_per_t
    

    return {"pos_interactions_plus":pos_interactions_plus, 
            "pos_interactions_minus":pos_interactions_minus, 
            "tot_interactions":tot_interactions,
            "T": T, "N": N, "edge_per_t": edge_per_t,
            "var_X_end": X[-1].var(),
            "skew_X_end": skew(X[-1]),
            "kurtosis_X_end": kurtosis(X[-1]),
            "bimodality_X_end": dipstat(X[-1]),
            }

########## all ##############
def epsilons_from_theta(parameters, dict_theta = False, numpy = False):
    
    sigmoid_fn = np_sigmoid if numpy else sigmoid
    if dict_theta:
        epsilon_plus = sigmoid_fn(parameters["theta0"]) / 2
        epsilon_minus = sigmoid_fn(parameters["theta1"]) / 2 + .5
        mu_plus = sigmoid_fn(parameters["theta2"]) / 10
        mu_minus = sigmoid_fn(parameters["theta3"]) / 10
        return epsilon_plus,epsilon_minus, mu_plus, mu_minus
    elif len(parameters.shape) == 1:
        epsilon_plus,epsilon_minus, mu_plus, mu_minus = sigmoid(parameters) / jnp.array([2,2,10,10]) + jnp.array([0.,.5,0.,0.])
        return epsilon_plus,epsilon_minus, mu_plus, mu_minus
    else:
        trans_theta = (sigmoid(parameters) / jnp.array([2,2,10,10]) + jnp.array([0.,.5,0.,0.])).T
        return trans_theta

def analyse_samples(samples):
        param_samples = epsilons_from_theta(samples, dict_theta = False, numpy = False).T
        param_mean, param_std = param_samples.mean(axis = 0), param_samples.std(axis = 0)
        
        return param_mean, param_std

def analyse_results(epsilon_plus, epsilon_minus, mu_plus, mu_minus, 
                    param_mean, param_std, n_samples, n_steps, n_simulations, id,
                    tot_time, method):
    params = np.array([epsilon_plus, epsilon_minus, mu_plus, mu_minus])

    param_names = ["epsilon_plus", "epsilon_minus", "mu_plus", "mu_minus"]
    out = {
            "id": id,
            "mse_params": ((params - param_mean)**2).mean().item(), 
            "mse_epsilon": ((params[:2] - param_mean[:2])**2).mean().item(), 
            "mse_mu": ((params[2:] - param_mean[2:])**2).mean().item(), 
            "tot_time": tot_time,
            "n_steps": n_steps, 
            "n_samples": n_samples,
            "method": method,
            "n_simulations": n_simulations
            }

    out.update({u + "_error": np.abs(params[k] - param_mean[k]) for k, u in enumerate(param_names)})
    out.update({u + "_mean": param_mean[k].item() for k, u in enumerate(param_names)})
    out.update({u + "_std": param_std[k].item() for k, u in enumerate(param_names)})
    out.update({u + "_real": params[k].item() for k, u in enumerate(param_names)})

    return out
        

def save_pickle(out, path):
    if path is not None:
        with open(path, "wb") as f:
            pickle.dump(out, f)


def complete_experiment(N, T, edge_per_t, rho = 32,
                        method = "svinormal",
                        epsilon_plus = None, epsilon_minus = None, mu_plus = None, mu_minus = None,
                        n_steps = 1000, n_samples = 100, populations_budget = 10, num_chains = 1,
                        intermediate_steps = None, intermediate_samples = None, warmup_samples = None, 
                        intermediate_populations = None, population_size = 200, 
                        lr = 0.01, progress_bar = False, timeout = 25000, id = None, date = None, save_data = True
                        ):
    if len(glob(f"../data/update_{date}/X_{id}*")) > 0:
        X_file = glob(f"../data/update_{date}/X_{id}*")[0]
        edges_file = glob(f"../data/update_{date}/edges_{id}*")[0]
        X = np.load(X_file)
        edges = np.load(edges_file)
        
        _,_,epsilon_plus,epsilon_minus, mu_plus, mu_minus = [int(u) for u in X_file.split("/")[-1].split("_")[2:-1]]
        epsilon_plus, epsilon_minus, mu_plus, mu_minus = np.array([epsilon_plus, epsilon_minus, mu_plus, mu_minus]) / 100
    else:
        if epsilon_plus is None:
            epsilon_plus = np.random.randint(5) * 0.1 + 0.05
            epsilon_minus = np.random.randint(5) * 0.1 + 0.55
            mu_plus = np.random.randint(5) * 0.02 + 0.01
            mu_minus = np.random.randint(5) * 0.02 + 0.01

        X, edges = BC_update.simulate_trajectory(N = N, T = T, edge_per_t = edge_per_t, 
                                                  epsilon_plus = epsilon_plus, epsilon_minus = epsilon_minus, 
                                                  mu_plus = mu_plus, mu_minus = mu_minus, rho = rho)  


        if save_data:
            np.save(f"../data/update_{date}/X_{id}_{int(epsilon_plus * 100)}_{int(epsilon_minus * 100)}_{int(mu_plus * 100)}_{int(mu_minus * 100)}_.npy", X)
            np.save(f"../data/update_{date}/edges_{id}_{int(epsilon_plus * 100)}_{int(epsilon_minus * 100)}_{int(mu_plus * 100)}_{int(mu_minus * 100)}_.npy", edges)
        

    analysis_data = count_interactions(X, edges)
    
    
    out = []
    if method == "svinormal":
        res_svinormal = train_svi(X, edges, guide_family = "normal", rho = rho,
             n_steps = n_steps, intermediate_steps = intermediate_steps, lr = lr, 
             progress_bar = progress_bar, id = id, timeout = timeout)
        out += res_svinormal
    if method == "sviNF":
        res_svinf = train_svi(X, edges, guide_family = "NF", rho = rho,
             n_steps = n_steps, intermediate_steps = intermediate_steps, lr = lr, 
             progress_bar = progress_bar, id = id, timeout = timeout)
        out += res_svinf
    if method == "mcmc":
        res_mcmc = train_mcmc(X, edges, intermediate_samples = intermediate_samples, warmup_samples = warmup_samples,  rho = rho,
                              n_samples = n_samples, num_chains = num_chains, progress_bar = progress_bar, id = id, timeout = timeout)
        out += res_mcmc
    if method == "abc":
        res_abc = train_abc(X, edges, populations_budget = populations_budget, intermediate_populations = intermediate_populations,
                            population_size = population_size, rho = rho, id = id, timeout = timeout)
        out += res_abc
    complete_analysis = [analyse_results(epsilon_plus, epsilon_minus, mu_plus, mu_minus, **res)|analysis_data for res in out]
    return complete_analysis
    # return out, analysis_data


if __name__ == '__main__':
    T, N = [int(sys.argv[k + 2]) for k in range(2)]
    method = sys.argv[4]
    date = sys.argv[5]

    rep = sys.argv[1]
    t0 = time()
    edge_per_t = 5

    id = f"{rep}_{N}_{T}"
    
    if not os.path.exists(f"../data/update_{date}"):
        try:
            os.mkdir(f"../data/update_{date}")
        except:
            None
    
    path = f"../data/update_{date}/estimation_T{T}_N{N}_rep{rep}_method{method}.pkl"

    print(f"++++++ update rep {rep} start {T} {N} {method} ++++++++")

    experiment = complete_experiment(N, T, edge_per_t, rho = 32,                    
                        n_steps=800, n_samples=200, 
                        method = method, populations_budget = 40, #intermediate_populations = 5,
                        population_size = 200, lr = 0.01,
                        id = id, date = date
                        )
    
    save_pickle(experiment, path)

    print(f">>>>>>>> update rep {rep} save {T} {N} {method} {round(time() - t0)}s <<<<<<<")
            
