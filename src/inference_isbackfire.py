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
import matplotlib.pyplot as plt



def compute_X_from_X0_params(X0, edges_iter, mu_plus, mu_minus, is_backfire = True):
    # edges_iter = (edges_t for edges_t in edges)
    # Xt = jax.lax.stop_gradient(X0.copy())
    Xt = X0.copy()
    X_list = [Xt.copy()]
    
    N = len(Xt)
    
    while True:
        edges_t = next(edges_iter, None)
        if edges_t is None:
            break
        
        u,v,s_plus,s_minus = edges_t.T
        u,v = u.astype(int),v.astype(int)
        diff_X = Xt[u] - Xt[v]

        updates_plus = mu_plus * s_plus * diff_X
        updates_minus = (mu_minus * s_minus * diff_X) * is_backfire
        # print(updates_minus)
        # Xt = Xt.at[v].add(updates_plus - updates_minus).clip(1e-5, 1 - 1e-5)
        Xt[v] += updates_plus - updates_minus
        Xt[v] = np.clip(Xt[v], 1e-5, 1 - 1e-5)
        
        X_list.append(Xt.copy())

    return jnp.stack(X_list)


def initialize_training(X, edges, mu_plus, mu_minus, rho = 32):
    T, N = X.shape    
    u,v,s_plus,s_minus,t = BC_leaders.convert_edges_uvst(edges)
    s_plus, s_minus = jnp.float32(s_plus), jnp.float32(s_minus)

    X0 = np.array(X[0])
    edges_iter = (edge for edge in edges)
    X_bc = compute_X_from_X0_params(X0, edges_iter, mu_plus, mu_minus, is_backfire = False)
    edges_iter = (edge for edge in edges)
    X_back = compute_X_from_X0_params(X0, edges_iter, mu_plus, mu_minus, is_backfire = True)
    u,v,t = u.astype(int), v.astype(int), t.astype(int)

    diff_X_bc = X_bc[t,u] - X_bc[t,v]
    diff_X_back = X_back[t,u] - X_back[t,v]

    return {"u": u, "v": v, "s_plus": s_plus, "s_minus": s_minus, "t": t,
            "N": N, "T": T, "rho": rho,
            "diff_X_bc": diff_X_bc, "diff_X_back": diff_X_back}

def model(data):
    dim = 3
    dist = distributions.Normal(jnp.zeros(dim), jnp.ones(dim)).to_event(1)
    params = numpyro.sample("theta", dist)
    
    theta = params[:2]
    param_backfire = params[2:]
    epsilon_plus, epsilon_minus = sigmoid(theta) /  2 + jnp.array([0.,.5])

    ############
    diff_X_bc,diff_X_back,u,v,s_plus, s_minus,t, rho, N, T = [data[k] for k in ["diff_X_bc", "diff_X_back","u","v",
                                                                      "s_plus", "s_minus","t",
                                                                      "rho", "N", "T"]]
    
    
    backfire_sample = numpyro.sample("backfire", distributions.RelaxedBernoulli(probs = param_backfire, temperature = jnp.array([0.1])).to_event(1))
    is_backfire = backfire_sample[0]
        
    s_plus = jnp.array(s_plus)
    s_minus = jnp.array(s_minus)
 
    diff_X = (1 - is_backfire) * diff_X_bc + is_backfire * diff_X_back
    kappas_plus = BC_leaders.kappa_plus_from_epsilon(epsilon_plus, diff_X, rho, with_jax = True)
    kappas_minus = BC_leaders.kappa_minus_from_epsilon(epsilon_minus, diff_X, rho, with_jax = True)
    kappas_ = jnp.concatenate([kappas_minus, kappas_plus])
    s = jnp.concatenate([s_minus, s_plus])

    with numpyro.plate("data", s.shape[0]):
        numpyro.sample("obs", distributions.Bernoulli(probs = kappas_), obs = s)

def train_svi(X, edges, mu_plus, mu_minus, guide_family = "normal", rho = 32,
              n_steps = 4000, intermediate_steps = None, lr = 0.01, 
              progress_bar = False, id = None, timeout = 3600):
    if intermediate_steps is None:
        intermediate_steps = n_steps
    
    if guide_family == "normal":
        guide = AutoNormal(model)
    if guide_family == "NF":
        guide = AutoBNAFNormal(model, num_flows = 1, hidden_factors = (8,8))
        n_steps = int(n_steps / 2)
        intermediate_steps = int(intermediate_steps / 2)
    
    data = initialize_training(jnp.array(X), jnp.array(edges), mu_plus, mu_minus, rho = rho)
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

        theta_samples = guide.sample_posterior(random.PRNGKey(0), svi_results.params, sample_shape = (200,))
        
        param_mean, param_std, backfire_mean, backfire_std = analyse_samples(theta_samples)
        
        res_analysis = {"param_mean": param_mean,
                        "param_std": param_std,
                        "backfire_mean": backfire_mean,
                        "backfire_std": backfire_std,
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


def train_mcmc(X, edges, mu_plus, mu_minus, intermediate_samples = None, rho = 32, num_chains = 1,
               warmup_samples = None, n_samples = 400, progress_bar = False, id = None, timeout = 3600):
    if intermediate_samples is None:
        intermediate_samples = n_samples
    if warmup_samples is None:
        warmup_samples = intermediate_samples

    data = initialize_training(jnp.array(X), jnp.array(edges), mu_plus, mu_minus, rho = rho)
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
        param_mean, param_std, backfire_mean, backfire_std = analyse_samples(mcmc_samples)
        res.append({"param_mean": param_mean,
                    "param_std": param_std,
                    "backfire_mean": backfire_mean, 
                    "backfire_std": backfire_std,
                    "tot_time": tot_time,
                    "n_simulations": None,
                    "method": "mcmc",
                    "n_steps": None,
                    "n_samples": intermediate_samples * (_ + 1),
                    "id": id})
        if tot_time > timeout:
            break

    return res


def create_summary_statistics(X0, edges_iter, edge_per_t, parameters, mu_plus, mu_minus, rho):
    summary_statistics_list = []
    Xt = X0.copy()
    N = len(Xt)
    
    while True:
        edges_t = next(edges_iter, None)
        if edges_t is None:
            break
        is_backfire = parameters["theta2"]
        epsilon_plus,epsilon_minus = epsilons_from_theta(parameters, dict_theta = True, numpy = True)
        u,v,_,_ = edges_t.T
        u,v = u.astype(int),v.astype(int)
        diff_X = Xt[u] - Xt[v]
        # s_plus = ((np.random.rand(edge_per_t) < np_sigmoid(rho * (epsilon_plus - np.abs(diff_X))))) + 0
        # s_minus = ((np.random.rand(edge_per_t) < np_sigmoid(-rho * (epsilon_minus - np.abs(diff_X))))) + 0
        s_plus =  (np.abs(diff_X) < epsilon_plus) + 0
        s_minus = (np.abs(diff_X) > epsilon_minus) + 0

        updates_plus = mu_plus * s_plus * diff_X 
        updates_minus = mu_minus * s_minus * diff_X * is_backfire
        Xt[v] += updates_plus - updates_minus
        Xt[v] = np.clip(Xt[v], 1e-5, 1 - 1e-5)
            
        summary_statistics_list.append(np.concatenate([u[None,:],v[None,:],s_plus[None,:], s_minus[None,:]])[None,:])

    edges_sim = np.concatenate(summary_statistics_list).transpose(0,2,1)
    return {"s_plus_sum": edges_sim[:,:,-2].sum(axis = 1), 
            "s_minus_sum": edges_sim[:,:,-1].sum(axis = 1)} 

def create_trajectory(X0, edges, parameters, mu_plus, mu_minus, rho):
    X0 = X0.copy()
    edges_iter = (edges_t for edges_t in edges)
    T, edge_per_t, _ = edges.shape
    summary_statistics = create_summary_statistics(X0, edges_iter, edge_per_t, parameters, mu_plus, mu_minus, rho)
    # summary_statistics = create_s_update_X(X0, edges_iter, edge_per_t, parameters, rho, [], [X0[None,:].copy()])
    return summary_statistics

def sim_trajectory_X0_edges(X0, edges, mu_plus, mu_minus, rho):
    return lambda parameters: create_trajectory(X0, edges, parameters, mu_plus, mu_minus, rho)

def train_abc(X, edges, mu_plus, mu_minus, populations_budget = 10, intermediate_populations = None,
              population_size = 200, rho = 32, id = None, timeout = 3600):
    if intermediate_populations is None:
        intermediate_populations = populations_budget
    
    T = len(X)
    res = []
    tot_time = 0
    model_abc = sim_trajectory_X0_edges(X[0], edges, mu_plus, mu_minus, rho)
    prior = pyabc.Distribution(
                theta0=pyabc.RV("norm", 0, 1),
                theta1=pyabc.RV("norm", 0, 1),
                theta2=pyabc.RV("rv_discrete", values = (np.arange(2), 0.5 * np.ones(2))))
    distance = pyabc.PNormDistance(2)
    obs = {"s_plus_sum": edges[:,:,-2].sum(axis = 1), 
           "s_minus_sum": edges[:,:,-1].sum(axis = 1)}
    abc = pyabc.ABCSMC(model_abc, prior, distance, population_size = population_size)#, sampler = SingleCoreSampler())
    db = "sqlite:///" + os.path.join(gettempdir(), f"{id}_is_backfire_test.db")
    history = abc.new(db, obs)
    run_id = history.id
    for _ in range(int(populations_budget / intermediate_populations)):
        abc_continued = pyabc.ABCSMC(model_abc, prior, distance, population_size = population_size)#, sampler = SingleCoreSampler())
        abc_continued.load(db, run_id)
        t0 = time()
        history = abc_continued.run(max_nr_populations = intermediate_populations,
                                    # minimum_epsilon = T,
                                    minimum_epsilon = 5 * (T ** (1/2)),
                                    max_walltime = timedelta(hours = 3))
        t1 = time()
        tot_time += (t1 - t0)
        theta_samples = jnp.array(history.get_distribution()[0])

        param_mean, param_std, backfire_mean, backfire_std = analyse_samples({"theta": theta_samples[:,:2], 
                                                                              "backfire": theta_samples[:,2]})
        res_analysis = {"param_mean": param_mean,
                        "param_std": param_std,
                        "backfire_mean": backfire_mean, 
                        "backfire_std": backfire_std,
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
        
        return epsilon_plus,epsilon_minus
    elif len(parameters.shape) == 1:
        epsilon_plus,epsilon_minus,is_backfire = sigmoid(parameters) / jnp.array([2, 2, 1]) + jnp.array([0.,.5, 0.])
        return epsilon_plus,epsilon_minus,is_backfire
    else:
        trans_theta = (sigmoid(parameters) / 2 + jnp.array([0.,.5])).T
        return trans_theta

def analyse_samples(samples):
        param_samples = epsilons_from_theta(samples["theta"][:,:2], dict_theta = False, numpy = False).T
        param_mean, param_std = param_samples.mean(axis = 0), param_samples.std(axis = 0)
        backfire_mean = samples["backfire"].mean(axis = 0)
        backfire_std = samples["backfire"].std(axis = 0)
        
        return param_mean, param_std, backfire_mean, backfire_std

def analyse_results(epsilon_plus, epsilon_minus, mu_plus, mu_minus, is_backfire, backfire_mean, backfire_std,
                    param_mean, param_std, n_samples, n_steps, n_simulations, id,
                    tot_time, method):
    params = np.array([epsilon_plus, epsilon_minus])

    param_names = ["epsilon_plus", "epsilon_minus"]
    
    out = {
            "id": id,
            "mse_epsilon": ((params[:2] - param_mean[:2])**2).mean().item(), 
            "tot_time": tot_time,
            "n_steps": n_steps, 
            "n_samples": n_samples,
            "method": method,
            "n_simulations": n_simulations,
            "mu_plus": mu_plus, 
            "mu_minus": mu_minus,
            "is_backfire": is_backfire,
            "round_backfire": backfire_mean.round().item(),
            "backfire_mean": backfire_mean.item(), 
            "backfire_std": backfire_std.item(),
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
                        lr = 0.01, progress_bar = True, timeout = 25000, id = None, date = None, save_data = True
                        ):
    if len(glob(f"../data/isbackfire_{date}/X_{id}*")) > 0:
        X_file = glob(f"../data/isbackfire_{date}/X_{id}*")[0]
        edges_file = glob(f"../data/isbackfire_{date}/edges_{id}*")[0]
        X = np.load(X_file)
        edges = np.load(edges_file)
        
        _,_,epsilon_plus,epsilon_minus, mu_plus, mu_minus, is_backfire = [int(u) for u in X_file.split("/")[-1].split("_")[2:-1]]
        epsilon_plus, epsilon_minus, mu_plus, mu_minus = np.array([epsilon_plus, epsilon_minus, mu_plus, mu_minus]) / 100
    else:
        if epsilon_plus is None:
            epsilon_plus = round(np.random.randint(5) * 0.1 + 0.05, 4)
            epsilon_minus = round(np.random.randint(5) * 0.1 + 0.55, 4)
            mu_plus = round(np.random.randint(10) * 0.02 + 0.01, 4)
            mu_minus = round(np.random.randint(10) * 0.02 + 0.01, 4)
            is_backfire = np.random.randint(2)
    
        X, edges = BC_update.simulate_trajectory(N = N, T = T, edge_per_t = edge_per_t, 
                                                  epsilon_plus = epsilon_plus, epsilon_minus = epsilon_minus, 
                                                  mu_plus = mu_plus, mu_minus = mu_minus * is_backfire, rho = rho)  


        if save_data:
            np.save(f"../data/isbackfire_{date}/X_{id}_{int(epsilon_plus * 100)}_{int(epsilon_minus * 100)}_{int(mu_plus * 100)}_{int(mu_minus * 100)}_{int(is_backfire)}_.npy", X)
            np.save(f"../data/isbackfire_{date}/edges_{id}_{int(epsilon_plus * 100)}_{int(epsilon_minus * 100)}_{int(mu_plus * 100)}_{int(mu_minus * 100)}_{int(is_backfire)}_.npy", edges)
        

    analysis_data = count_interactions(X, edges)
    
    
    out = []
    if method == "svinormal":
        res_svinormal = train_svi(X, edges, mu_plus = mu_plus, mu_minus = mu_minus, guide_family = "normal", rho = rho,
             n_steps = n_steps, intermediate_steps = intermediate_steps, lr = lr, 
             progress_bar = progress_bar, id = id, timeout = timeout)
        out += res_svinormal
    if method == "sviNF":
        res_svinf = train_svi(X, edges, mu_plus = mu_plus, mu_minus = mu_minus, guide_family = "NF", rho = rho,
             n_steps = n_steps, intermediate_steps = intermediate_steps, lr = lr, 
             progress_bar = progress_bar, id = id, timeout = timeout)
        out += res_svinf
    if method == "mcmc":
        res_mcmc = train_mcmc(X, edges, mu_plus = mu_plus, mu_minus = mu_minus, intermediate_samples = intermediate_samples, warmup_samples = warmup_samples,  rho = rho,
                              n_samples = n_samples, num_chains = num_chains, progress_bar = progress_bar, id = id, timeout = timeout)
        out += res_mcmc
    if method == "abc":
        res_abc = train_abc(X, edges, mu_plus = mu_plus, mu_minus = mu_minus, populations_budget = populations_budget, intermediate_populations = intermediate_populations,
                            population_size = population_size, rho = rho, id = id, timeout = timeout)
        out += res_abc
    complete_analysis = [analyse_results(epsilon_plus, epsilon_minus, mu_plus, mu_minus, is_backfire, **res)|analysis_data for res in out]
    return complete_analysis
    # return out, analysis_data



if __name__ == '__main__':
    T, N = [int(sys.argv[k + 2]) for k in range(2)]
    method = sys.argv[4]
    rep = sys.argv[1]
    date = sys.argv[5]
    
    edge_per_t = 10
    

    id = f"{rep}_{N}_{T}"
    
    if not os.path.exists(f"../data/isbackfire_{date}"):
        try:
            os.mkdir(f"../data/isbackfire_{date}")
        except:
            None
    path = f"../data/isbackfire_{date}/estimation_T{T}_N{N}_rep{rep}_method{method}.pkl"
    
        
    print(f"++++++ isbackfire rep {rep} start T{T} N{N} {method} ++++++ ")

    # if ((N / initial_leaders_ratio) < 1)|((T > 2048) & (method in ["abc", "mcmc"]))|((N > 200) & (method == "sviNF")):
    
    t0 = time()
    experiment = complete_experiment(N, T, edge_per_t, rho = 32,
                                    n_steps = 20000,
                                    n_samples = 800, method = method, #intermediate_populations = 5,
                                    populations_budget = 40, population_size = 5000, lr = 0.01,
                                    id = id, date = date
                                    )
        
    save_pickle(experiment, path)
    print(f">>>>>>>> rep {rep} save {T} {N} {method} {round(time() - t0)}s<<<<<<<")





