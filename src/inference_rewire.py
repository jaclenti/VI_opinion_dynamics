import sys 
sys.path += ["../src"]
import BC_feed, BC_leaders, BC_rewire
import jax, jaxlib
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import jax.random as random
import numpyro
from time import time
import os
from tempfile import gettempdir
from jax.scipy.special import expit as sigmoid
from numpyro.infer import SVI, Trace_ELBO, TraceGraph_ELBO, MCMC, NUTS
from numpyro.infer.autoguide import AutoNormal, AutoBNAFNormal
from pyro.nn import PyroModule
from numpyro import distributions
from numpyro.optim import Adam
from time import time
import pickle
from glob import glob
from pyABC_ import pyabc
from pyABC_.pyabc.sampler import SingleCoreSampler
from scipy.special import expit as np_sigmoid
from datetime import timedelta
from diptest import dipstat
from scipy.stats import kurtosis, skew

# numpyro.set_platform('cpu')


def count_interactions(X, edges, q):
    T, N = X.shape
    _,edge_per_t,_ = edges.shape
    
    pos_interactions_plus, pos_interactions_minus, pos_rewires = edges[:,:,2].sum(), edges[:,:,3].sum(), edges[:,:,4].sum()
    tot_interactions = (T - 1) * edge_per_t
    
    return {"pos_interactions_plus":pos_interactions_plus, "pos_interactions_minus":pos_interactions_minus, "pos_rewires":pos_rewires, 
            "tot_interactions":tot_interactions, "q":q,
            "T": T, "N": N, "edge_per_t": edge_per_t,
                        "var_X_end": X[-1].var(),
            "skew_X_end": skew(X[-1]),
            "kurtosis_X_end": kurtosis(X[-1]),
            "bimodality_X_end": dipstat(X[-1]),
}


def analyse_results(epsilon_plus, epsilon_minus, beta, param_mean, param_std,
                    tot_time, n_steps, n_samples, n_simulations, method, id):
    params = np.array([epsilon_plus, epsilon_minus, beta])

    param_names = ["epsilon_plus", "epsilon_minus", "beta"]
    
    out = {
            "id": id,
            "mse_params": ((params - param_mean)**2).mean().item(), 
            "mse_epsilon": ((params[:2] - param_mean[:2])**2).mean().item(), 
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


def initialize_training(X, edges, rho_up = 32, rho_lr = 4):
    uvst = jnp.array(BC_feed.convert_edges_uvst(edges))
    u,v,s_plus,s_minus,r,up,t = uvst
    u_up,v_up,s_plus_up,s_minus_up,t_up = u[up==1],v[up==1],s_plus[up==1],s_minus[up==1],t[up==1]
    u_lr,v_lr,r_lr,t_lr = u[up==0],v[up==0],r[up==0],t[up==0]
    s_plus_up,s_minus_up,r_lr = jnp.float32(s_plus_up),jnp.float32(s_minus_up),jnp.float32(r_lr)

    diff_X_up = jnp.abs(X[t_up,u_up] - X[t_up,v_up])
    diff_X_lr = jnp.abs(X[t_lr,u_lr] - X[t_lr,v_lr])

    
    return {"s_plus_up":s_plus_up,"s_minus_up":s_minus_up,"r_lr":r_lr,
            "diff_X_up":diff_X_up,"diff_X_lr":diff_X_lr,
            "rho_up":rho_up,"rho_lr":rho_lr}


def model(data):
    s_plus_up,s_minus_up,r_lr,diff_X_up,diff_X_lr,rho_up,rho_lr = [data[k] for k in ["s_plus_up","s_minus_up","r_lr","diff_X_up","diff_X_lr","rho_up","rho_lr"]]
    
    dim = 3
    dist = distributions.Normal(jnp.zeros(dim), jnp.ones(dim)).to_event(1)
    params = numpyro.sample("theta", dist)

    epsilon_plus, epsilon_minus, beta = epsilons_from_theta(params)
    
    kappas_plus = BC_leaders.kappa_plus_from_epsilon(epsilon_plus, diff_X_up, rho_up, with_jax = True)
    kappas_minus = BC_leaders.kappa_minus_from_epsilon(epsilon_minus, diff_X_up, rho_up, with_jax = True)
    kappas_beta = BC_leaders.kappa_minus_from_epsilon(beta, diff_X_lr, rho_lr, with_jax = True)
    

    kappas_ = jnp.concatenate([kappas_minus, kappas_plus, kappas_beta])
    s = jnp.concatenate([s_minus_up, s_plus_up, r_lr])

    with numpyro.plate("data", s.shape[0]):
        numpyro.sample("obs", distributions.Bernoulli(probs = kappas_), obs = s)


def epsilons_from_theta(parameters, dict_theta = False, numpy = False):
    
    sigmoid_fn = np_sigmoid if numpy else sigmoid
    if dict_theta:
        epsilon_plus = sigmoid_fn(parameters["theta0"]) / 2
        epsilon_minus = sigmoid_fn(parameters["theta1"]) / 2 + .5
        beta = sigmoid_fn(parameters["theta2"])
        
        return epsilon_plus,epsilon_minus, beta
    elif len(parameters.shape) == 1:
        epsilon_plus,epsilon_minus, beta = sigmoid(parameters) / jnp.array([2,2,1]) + jnp.array([0.,.5,0.])
        return epsilon_plus,epsilon_minus, beta
    else:
        trans_theta = (sigmoid(parameters) / jnp.array([2,2,1]) + jnp.array([0.,.5,0.])).T
        return trans_theta

def analyse_samples(samples):
        param_samples = epsilons_from_theta(samples, dict_theta = False, numpy = False).T
        param_mean, param_std = param_samples.mean(axis = 0), param_samples.std(axis = 0)
        
        return param_mean, param_std

def train_svi(X, edges, n_steps = 1000, rho_up = 32, rho_lr = 4, lr = 0.01,
              intermediate_steps = None, progress_bar = False, 
              guide_family = "normal", id = None, timeout = 3600):
    if intermediate_steps is None:
        intermediate_steps = n_steps
    
    if guide_family == "normal":
        guide = AutoNormal(model)
    if guide_family == "NF":
        guide = AutoBNAFNormal(model, num_flows = 2)
        n_steps = int(n_steps / 2)
        intermediate_steps = int(intermediate_steps / 2)
    
    data = initialize_training(jnp.array(X), jnp.array(edges), rho_up = rho_up, rho_lr = rho_lr)
    optimizer = Adam(step_size = lr)
    svi = SVI(model, guide, optimizer, loss = TraceGraph_ELBO())
    res = []
    last_state = None

    tot_time = 0
    
    for _ in range(int(n_steps / intermediate_steps)):
        t0 = time()
        svi_results = svi.run(random.PRNGKey(np.random.randint(low = 0, high = 10**7)), intermediate_steps, data, init_state = last_state, progress_bar = progress_bar)
        t1 = time()
        tot_time += t1 - t0

        theta_samples = guide.sample_posterior(random.PRNGKey(np.random.randint(low = 0, high = 10**7)), svi_results.params, sample_shape = (200,))["theta"]
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

        


def train_mcmc(X, edges, n_samples = 400, rho_up = 32, rho_lr = 4, intermediate_samples = None, num_chains = 1, 
               warmup_samples = None, progress_bar = False, id = None, timeout = 3600):
    if intermediate_samples is None:
        intermediate_samples = n_samples
    if warmup_samples is None:
        warmup_samples = intermediate_samples

    data = initialize_training(jnp.array(X), jnp.array(edges), rho_up = rho_up, rho_lr = rho_lr)

    mcmc = MCMC(NUTS(model), num_warmup = warmup_samples, num_chains = num_chains, num_samples = intermediate_samples, progress_bar = progress_bar)
    key = random.PRNGKey(np.random.randint(low = 0, high = 10**7))
    
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

##### pyabc #####
# def create_s_update_X(X_iter, edges_iter, edge_per_t, parameters, q, rho_up, rho_lr,
#                      summary_statistics_list):
#     edges_t = next(edges_iter, None)
#     X_t = next(X_iter, None)
#     if edges_t is not None:
#         epsilon_plus,epsilon_minus, beta = epsilons_from_theta(parameters, dict_theta = True, numpy = True)
#         u,v,_,_,_,is_update = edges_t.T
#         try:
#             diff_X = X_t[u] - X_t[v]
#         except:
#             print(v)
            
#         s_plus = is_update * ((np.random.rand(edge_per_t) < np_sigmoid(rho_up * (epsilon_plus - np.abs(diff_X)))))
#         s_minus = is_update * ((np.random.rand(edge_per_t) < np_sigmoid(-rho_up * (epsilon_minus - np.abs(diff_X)))))
#         s_lr = (1 - is_update) * ((np.random.rand(edge_per_t) < np_sigmoid(-rho_lr * (beta - np.abs(diff_X)))))

#         summary_statistics_list.append(np.concatenate([s_plus[None,:], s_minus[None,:],s_lr[None,:]])[None,:])
#         create_s_update_X(X_iter, edges_iter, edge_per_t, parameters, q, rho_up, rho_lr, summary_statistics_list)
#     edges_sim = np.concatenate(summary_statistics_list).transpose(0,2,1)
#     return {"s_plus_sum": (edges_sim[:,:,-3]).sum(axis = 1), 
#            "s_minus_sum": (edges_sim[:,:,-2]).sum(axis = 1),
#            "s_lr_sum": (edges_sim[:,:,-1]).sum(axis = 1),
#            }

def create_summary_statistics(X0, edges_iter, edge_per_t, parameters, q, rho_up, rho_lr, mu_plus, mu_minus):
    summary_statistics_list = []
    Xt = X0.copy()
    N = len(Xt)
    
    while True:
        edges_t = next(edges_iter, None)
        if edges_t is None:
            break
        epsilon_plus,epsilon_minus, beta = epsilons_from_theta(parameters, dict_theta = True, numpy = True)
        u,v,_,_,_,is_update = edges_t.T
        u,v = u.astype(int),v.astype(int)
        diff_X = Xt[u] - Xt[v]
            
        # s_plus = is_update * ((np.random.rand(edge_per_t) < np_sigmoid(rho_up * (epsilon_plus - np.abs(diff_X)))))
        # s_minus = is_update * ((np.random.rand(edge_per_t) < np_sigmoid(-rho_up * (epsilon_minus - np.abs(diff_X)))))
        # s_lr = (1 - is_update) * ((np.random.rand(edge_per_t) < np_sigmoid(-rho_lr * (beta - np.abs(diff_X)))))
        s_plus = is_update * ((np.abs(diff_X) < epsilon_plus) + 0.)
        s_minus = is_update * ((np.abs(diff_X) > epsilon_minus) + 0.)
        s_lr = (1 - is_update) * ((np.random.rand(edge_per_t) < np_sigmoid(-rho_lr * (beta - np.abs(diff_X)))))

        updates_plus = mu_plus * s_plus * is_update * diff_X
        updates_minus = mu_minus * s_plus * is_update * diff_X
        
        Xt[v] += updates_plus - updates_minus
        Xt[v] = np.clip(Xt[v], 1e-5, 1 - 1e-5)

        summary_statistics_list.append(np.concatenate([s_plus[None,:], s_minus[None,:],s_lr[None,:]])[None,:])
            
    edges_sim = np.concatenate(summary_statistics_list).transpose(0,2,1)

    return {"s_plus_sum": edges_sim[:,:,-3].sum(axis = 1), 
            "s_minus_sum": edges_sim[:,:,-2].sum(axis = 1),
            "s_lr_sum": (edges_sim[:,:,-1]).sum(axis = 1),}


def create_trajectory(X, edges, parameters, q, rho_up, rho_lr, mu_plus, mu_minus):
    edges_iter = (edges_t for edges_t in edges)
    X0 = X[0]
    T, edge_per_t, _ = edges.shape

    summary_statistics = create_summary_statistics(X0, edges_iter, edge_per_t, parameters, q, rho_up, rho_lr, mu_plus, mu_minus)
    # summary_statistics = create_s_update_X(X_iter, edges_iter, edge_per_t, parameters, q, rho_up, rho_lr, [])
    return summary_statistics

def sim_trajectory_X0_edges(X, edges, q, rho_up, rho_lr, mu_plus, mu_minus):
    return lambda parameters: create_trajectory(X, edges, parameters, q, rho_up, rho_lr, mu_plus, mu_minus)



def train_abc(X, edges, q, rho_up = 32, rho_lr = 4, populations_budget = 10, intermediate_populations = None,
              population_size = 200, id = None, timeout = 3600, mu_plus = 0.02, mu_minus = 0.02):
    if intermediate_populations is None:
        intermediate_populations = populations_budget
    
    T = len(X)
    res = []
    tot_time = 0
    model_abc = sim_trajectory_X0_edges(X, edges, q, rho_up, rho_lr, mu_plus, mu_minus)
    prior = pyabc.Distribution(
                theta0=pyabc.RV("norm", 0, 1),
                theta1=pyabc.RV("norm", 0, 1),
                theta2=pyabc.RV("norm", 0, 1))
                
    distance = pyabc.PNormDistance(2)
    obs = {"s_plus_sum": (edges[:,:,-4] * edges[:,:,-1]).sum(axis = 1), 
           "s_minus_sum": (edges[:,:,-3] * edges[:,:,-1]).sum(axis = 1),
           "s_lr_sum": (edges[:,:,-2] * (1 - edges[:,:,-1])).sum(axis = 1),
           }
    abc = pyabc.ABCSMC(model_abc, prior, distance, population_size = population_size)#, sampler = SingleCoreSampler())
    db = "sqlite:///" + os.path.join(gettempdir(), f"{id}_rewire_test.db")
    history = abc.new(db, obs)
    run_id = history.id
    for _ in range(int(populations_budget / intermediate_populations)):
        abc_continued = pyabc.ABCSMC(model_abc, prior, distance, population_size = population_size)#, sampler = SingleCoreSampler())
        abc_continued.load(db, run_id)
        t0 = time()
        history = abc_continued.run(max_nr_populations = intermediate_populations,
                                    minimum_epsilon = 5 * (T ** (1 / 2)),
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




def complete_experiment(N, T, edge_per_t, q = 0.5, rho_up = 32, rho_lr = 4,
                        epsilon_plus = None, epsilon_minus = None, beta = None, mu_plus = 0.02, mu_minus = 0.02,
                        n_steps = 4000, n_samples = 40, populations_budget = 10, 
                        intermediate_steps = None, intermediate_samples = None, warmup_samples = None, num_chains = 1,
                        intermediate_populations = None, population_size = 200, method = "svinormal",
                        lr_svi = 0.01, progress_bar = False, timeout = 25000, id = None, date = None, save_data = True
                        ):
    if len(glob(f"../data/rewire_{date}/X_{id}*")) > 0:
        X_file = glob(f"../data/rewire_{date}/X_{id}*")[0]
        edges_file = glob(f"../data/rewire_{date}/edges_{id}*")[0]
        X = np.load(X_file)
        edges = np.load(edges_file)
        
        _,_,epsilon_plus,epsilon_minus, beta = [int(u) for u in X_file.split("/")[-1].split("_")[2:-1]]
        epsilon_plus, epsilon_minus, beta = np.array([epsilon_plus, epsilon_minus, beta]) / 100
    else:
        if epsilon_plus is None:
            epsilon_plus = np.random.randint(5) * 0.1 + 0.05
            epsilon_minus = np.random.randint(5) * 0.1 + 0.55
            beta = np.random.randint(10) * 0.1 + 0.05
        
        X, edges = BC_rewire.simulate_trajectory(N = N, T = T, edge_per_t = edge_per_t, epsilon_plus = epsilon_plus, epsilon_minus = epsilon_minus, 
                                                beta = beta, q = q, mu_plus = mu_plus, mu_minus = mu_minus, 
                                                rho_up = rho_up, rho_lr = rho_lr)
        if save_data:
            np.save(f"../data/rewire_{date}/X_{id}_{int(epsilon_plus * 100)}_{int(epsilon_minus * 100)}_{int(beta * 100)}_.npy", X)
            np.save(f"../data/rewire_{date}/edges_{id}_{int(epsilon_plus * 100)}_{int(epsilon_minus * 100)}_{int(beta * 100)}_.npy", edges)
        
        
    analysis_data = count_interactions(X, edges, q)

    out = []
    if method == "svinormal":
        res_svinormal = train_svi(X, edges, n_steps = n_steps, rho_up = rho_up, rho_lr = rho_lr, intermediate_steps = intermediate_steps,
                                  guide_family = "normal", lr = lr_svi, progress_bar = progress_bar, id = id, timeout = timeout)
        out += res_svinormal
        
    if method == "sviNF":
        res_svinormal = train_svi(X, edges, n_steps = n_steps, rho_up = rho_up, rho_lr = rho_lr, intermediate_steps = intermediate_steps,
                                  guide_family = "NF", lr = lr_svi, progress_bar = progress_bar, id = id, timeout = timeout)
        out += res_svinormal
        
    if method == "mcmc":
        res_mcmc = train_mcmc(X, edges, n_samples = n_samples, rho_up = rho_up, rho_lr = rho_lr, intermediate_samples = intermediate_samples, num_chains = num_chains,
                              warmup_samples = warmup_samples, progress_bar = progress_bar, id = id, timeout = timeout)
        out += res_mcmc

    if method == "abc":
        res_abc = train_abc(X, edges, q, rho_up = rho_up, rho_lr = rho_lr, populations_budget = populations_budget, intermediate_populations = intermediate_populations,
                            population_size = population_size, id = id, timeout = timeout, mu_plus = mu_plus, mu_minus = mu_minus)
        out += res_abc
    
    complete_analysis = [analyse_results(epsilon_plus, epsilon_minus, beta, **res)|analysis_data for res in out]
    return complete_analysis


if __name__ == '__main__':
    T, Q = [int(sys.argv[k + 2]) for k in range(2)]
    q = Q / 10
    method = sys.argv[4]
    date = sys.argv[5]

    rep = sys.argv[1]
    t0 = time()
    edge_per_t = 10

    id = f"{rep}_{Q}_{T}"
    
    path = f"../data/rewire_{date}/estimation_Q{Q}_T{T}_rep{rep}_method{method}.pkl"

    if not os.path.exists(f"../data/rewire_{date}"):
        try:
            os.mkdir(f"../data/rewire_{date}")
        except:
            None
    print(f"++++++ rewire rep {rep} start {Q} {T} {method} ++++++++")

    experiment = complete_experiment(400, T, edge_per_t,  q = q, rho_up = 32, rho_lr = 4,
                        n_steps = 20000, n_samples = 600, 
                        method = method, populations_budget = 40,
                        population_size = 200, lr_svi = 0.01, #intermediate_populations = 5,
                        id = id, date = date
                        )
                        
    
    save_pickle(experiment, path)

    print(f">>>>>>>> rewire rep {rep} save {Q} {T} {method} {round(time() - t0)}s <<<<<<<")
            

