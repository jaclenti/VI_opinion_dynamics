import sys 
sys.path += ["../src"]
import BC_feed, BC_leaders
import jax, jaxlib
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import jax.random as random
from pyABC_.pyabc.sampler import SingleCoreSampler
import numpyro
from time import time
from jax.scipy.special import expit as sigmoid
from numpyro.infer import SVI, Trace_ELBO, TraceGraph_ELBO, MCMC, NUTS
from numpyro.infer.autoguide import AutoNormal, AutoBNAFNormal
import os
from tempfile import gettempdir
from numpyro import distributions
from numpyro.optim import Adam
from time import time
import pickle
from scipy.special import expit as np_sigmoid
from glob import glob
from pyABC_ import pyabc
import pandas as pd
from datetime import timedelta
from diptest import dipstat
from scipy.stats import kurtosis, skew
           
sys.setrecursionlimit(10000)

# numpyro.set_platform('cpu')


def epsilons_from_theta(parameters, dict_theta = False, numpy = False):
    
    sigmoid_fn = np_sigmoid if numpy else sigmoid
    if dict_theta:
        epsilon_plus = sigmoid_fn(parameters["theta0"]) / 2
        epsilon_minus = sigmoid_fn(parameters["theta1"]) / 2 + .5
        obs_f = parameters["thetaf"] + 1
        return epsilon_plus,epsilon_minus, obs_f
    elif len(parameters.shape) == 1:
        epsilon_plus,epsilon_minus = sigmoid(parameters[:2]) / 2 + jnp.array([0.,.5])
        obs_f = parameters[2] + 1
        return epsilon_plus,epsilon_minus, obs_f
    else:
        trans_epsilon = (sigmoid(parameters[:,:2]) / 2 + jnp.array([0.,.5])).T
        obs_f = parameters[:,2] + 1
        trans_theta = jnp.concatenate([trans_epsilon, obs_f[None,:]], axis = 0)
        return trans_theta


def count_interactions(X, edges, max_f_possible):
    T, N = X.shape
    _,edge_per_t,_ = edges.shape
    
    pos_interactions_plus, pos_interactions_minus = edges[:,:,-2].sum(), edges[:,:,-1].sum()
    tot_interactions = (T - 1) * edge_per_t
    
    return {"pos_interactions_plus":pos_interactions_plus, "pos_interactions_minus":pos_interactions_minus, 
            "tot_interactions":tot_interactions,
            "T": T, "N": N, "edge_per_t": edge_per_t, 
            "max_f_possible": max_f_possible,
            "var_X_end": X[-1].var(),
            "skew_X_end": skew(X[-1]),
            "kurtosis_X_end": kurtosis(X[-1]),
            "bimodality_X_end": dipstat(X[-1]),
}

def analyse_results(epsilon_plus, epsilon_minus, obs_f, max_f_possible, argmax_samples_feed, 
                    feed_mean_est, epsilon_mean, epsilon_std,
                    tot_time, n_steps, n_samples, n_simulations, id, method):
    epsilons = jnp.array([epsilon_plus, epsilon_minus])
    
    epsilon_names = ["epsilon_plus", "epsilon_minus"]
    out = {"id": id,
            "method": method,
            "mse_epsilon": ((epsilons - epsilon_mean[:2])**2).mean().item(), 
            "correct_f": obs_f == argmax_samples_feed + 0.,
            "abs_error_f": jnp.abs(obs_f - argmax_samples_feed).item(),
            "rel_error_f": (jnp.abs(obs_f - argmax_samples_feed) / max_f_possible).item(),
            "obs_f": obs_f,
            "max_f_possible": max_f_possible,
            "argmax_samples_feed": argmax_samples_feed,
            "feed_mean_est": feed_mean_est,
            "tot_time": tot_time,
            "n_steps": n_steps, 
            "n_samples": n_samples,
            "n_simulations": n_simulations
            }

    out.update({u + "_error": np.abs(epsilons[k] - epsilon_mean[k]) for k, u in enumerate(epsilon_names)})
    out.update({u + "_mean": epsilon_mean[k].item() for k, u in enumerate(epsilon_names)})
    out.update({u + "_std": epsilon_std[k].item() for k, u in enumerate(epsilon_names)})
    out.update({u + "_real": epsilons[k].item() for k, u in enumerate(epsilon_names)})

    return out
        

def save_pickle(out, path):
    if path is not None:
        with open(path, "wb") as f:
            pickle.dump(out, f)


def initialize_training(X, edges, max_f_possible, rho = 32):
    T, N = X.shape
    _,edge_per_t,_ = edges.shape
    
    uvst = BC_feed.convert_edges_uvst(edges)
    max_u = uvst[:-4,:]
    v = uvst[-4,:]
    s_plus = uvst[-3,:]
    s_minus = uvst[-2,:]
    t = uvst[-1,:]

    s_plus = jnp.float32(s_plus)
    s_minus = jnp.float32(s_minus)
    X_t_mean_obsu = X[t, max_u].cumsum(axis = 0) / (jnp.arange(max_f_possible)[:,None] + 1.)
    possible_diff_X = jnp.abs(X[t,v][None,:] - X_t_mean_obsu)

    
    return {"possible_diff_X": possible_diff_X,"u": max_u, "max_f_possible": max_f_possible,
             "v": v,"s_plus": s_plus, "s_minus": s_minus,"t": t, "rho": rho}

def model(data):
    possible_diff_X,max_u, max_f_possible, v,s_plus, s_minus,t, rho = [data[k] for k in ["possible_diff_X","u", "max_f_possible", "v","s_plus", "s_minus","t", "rho"]]
    dim = max_f_possible + 2
    dist = distributions.Normal(jnp.zeros(dim), jnp.ones(dim)).to_event(1)
    params = numpyro.sample("theta", dist)
    
    theta = params[:2]
    epsilon_plus, epsilon_minus = sigmoid(theta) / 2 + jnp.array([0.,.5])
        
    logits = params[2:]
    
    key = random.PRNGKey(np.random.randint(low = 0, high = 10**10))
    y = logits + distributions.Gumbel().sample(key = key, sample_shape = logits.shape)
    one_hot_feed = jax.nn.softmax(y / 0.2) # divide by the temperature
    
    possible_kappas_plus = BC_leaders.kappa_plus_from_epsilon(epsilon_plus, possible_diff_X, rho, with_jax = True)
    possible_kappas_minus = BC_leaders.kappa_minus_from_epsilon(epsilon_minus, possible_diff_X, rho, with_jax = True)
    
    kappas_plus = one_hot_feed @ possible_kappas_plus 
    kappas_minus = one_hot_feed @ possible_kappas_minus 
        
    kappas_ = jnp.concatenate([kappas_minus, kappas_plus])
    s = jnp.concatenate([s_minus, s_plus])

    with numpyro.plate("data", s.shape[0]):
        numpyro.sample("obs", distributions.Bernoulli(probs = kappas_), obs = s)

# def analyse_samples(samples, max_f_possible):
#     epsilon_samples = epsilons_from_theta(samples[:,:2], dict_theta = False, numpy = False).T
#     epsilon_mean, epsilon_std = epsilon_samples.mean(axis = 0), epsilon_samples.std(axis = 0)
#     argmax_samples_feed = jax.scipy.stats.mode(samples[:,2:].argmax(axis = 1)).mode.item() + 1
#     sig_samples_mean = sigmoid(samples[:,2:]).mean(axis = 0)
#     print(sig_samples_mean)
#     samples_prob = sig_samples_mean / (sig_samples_mean.sum())
#     print(samples_prob)
#     feed_mean_est = (samples_prob * (jnp.arange(max_f_possible) + 1)).sum().item()
#     print(feed_mean_est)
        
#     return epsilon_mean, epsilon_std, argmax_samples_feed,feed_mean_est

def analyse_samples(samples, gumbel_softmax = True):
    epsilon_samples = epsilons_from_theta(samples[:,:2], dict_theta = False, numpy = False).T
    epsilon_mean, epsilon_std = epsilon_samples.mean(axis = 0), epsilon_samples.std(axis = 0)
    if gumbel_softmax:
        max_f_possible = samples.shape[1] - 2
        mode_samples_feed = jax.scipy.stats.mode(samples[:,2:].argmax(axis = 1)).mode.item() + 1
        sig_samples_mean = sigmoid(samples[:,2:]).mean(axis = 0)
        samples_prob = sig_samples_mean / (sig_samples_mean.sum())
        mean_samples_feed = (samples_prob * (jnp.arange(max_f_possible) + 1)).sum().item()
    else:
        mode_samples_feed = jax.scipy.stats.mode(samples[:,2]).mode.item() + 1
        mean_samples_feed = samples[:,2].mean().item() + 1
        
    return epsilon_mean, epsilon_std, mode_samples_feed,mean_samples_feed



def train_svi(X, edges, max_f_possible, n_steps, intermediate_steps = None, 
              progress_bar = False, lr = 0.01, rho = 32, timeout = 3600,
              guide_family = "normal", id = None):
    if intermediate_steps is None:
        intermediate_steps = n_steps
    
    if guide_family == "normal":
        guide = AutoNormal(model)
    if guide_family == "NF":
        guide = AutoBNAFNormal(model, num_flows = 2)
        n_steps = int(n_steps / 2)
        intermediate_steps = int(intermediate_steps / 2)
        # guide = AutoIAFNormal(model)
        # n_steps = int(n_steps / 5)
        # intermediate_steps = int(intermediate_steps / 5)

    data = initialize_training(X, edges, max_f_possible, rho = rho)
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

       
        svi_samples = guide.sample_posterior(random.PRNGKey(2), svi_results.params, sample_shape = (200,))
        epsilon_mean, epsilon_std, argmax_samples_feed,feed_mean_est = analyse_samples(svi_samples["theta"])#, max_f_possible)
        
        res.append({"method": "svi" + guide_family,
                    "id": id,
                    "epsilon_mean": epsilon_mean, 
                    "epsilon_std": epsilon_std, 
                    "argmax_samples_feed": argmax_samples_feed,
                    "feed_mean_est": feed_mean_est,
                    "tot_time": tot_time,
                    "n_simulations": None,
                    "n_steps": intermediate_steps * (_ + 1),
                    "n_samples": None,
                    })
        last_state = svi_results.state

        if tot_time > timeout:
            break

    return res



def train_mcmc(X, edges, max_f_possible, n_samples, intermediate_samples = None, rho = 32, num_chains = 1,
               warmup_samples = None, progress_bar = False, id = None, timeout = 3600):
    if intermediate_samples is None:
        intermediate_samples = n_samples
    if warmup_samples is None:
        warmup_samples = intermediate_samples

    data = initialize_training(X, edges, max_f_possible, rho = rho)

    mcmc = MCMC(NUTS(model), num_warmup = warmup_samples, num_samples = intermediate_samples, num_chains = num_chains, progress_bar = progress_bar)
    key = random.PRNGKey(0)
    
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
        epsilon_mean, epsilon_std, argmax_samples_feed,feed_mean_est = analyse_samples(mcmc_samples["theta"])#, max_f_possible)
        
        res.append({"id": id,
                    "method": "mcmc",
                    "epsilon_mean": epsilon_mean, 
                    "epsilon_std": epsilon_std, 
                    "argmax_samples_feed": argmax_samples_feed,
                    "feed_mean_est": feed_mean_est,
                    "tot_time": tot_time,
                    "n_simulations": None,
                    "n_steps": None,
                    "n_samples": intermediate_samples * (_ + 1),
                    })
        if tot_time > timeout:
            break

    return res


######## pyabc #############
def create_summary_statistics(X0, edges_iter, edge_per_t, parameters, rho, mu_plus, mu_minus):
    summary_statistics_list = []
    Xt = X0.copy()
    N = len(Xt)
    
    while True:
        edges_t = next(edges_iter, None)
        if edges_t is None:
            break
        epsilon_plus,epsilon_minus, obs_f = epsilons_from_theta(parameters, dict_theta = True, numpy = True)
        u = edges_t.T[:obs_f,:]
        v = edges_t.T[-2,:]
        
        diff_X = Xt[u].mean(axis = 0) - Xt[v]

        s_plus =  (np.abs(diff_X) < epsilon_plus) + 0
        s_minus = (np.abs(diff_X) > epsilon_minus) + 0

        # s_plus = ((np.random.rand(edge_per_t) < np_sigmoid(rho * (epsilon_plus - np.abs(diff_X))))) + 0
        # s_minus = ((np.random.rand(edge_per_t) < np_sigmoid(-rho * (epsilon_minus - np.abs(diff_X))))) + 0

        updates_plus = mu_plus * s_plus * diff_X 
        updates_minus = mu_minus * s_minus * diff_X 
        Xt[v] += updates_plus - updates_minus
        Xt[v] = np.clip(Xt[v], 1e-5, 1 - 1e-5)
        summary_statistics_list.append(np.concatenate([s_plus[None,:], s_minus[None,:]])[None,:])
    edges_sim = np.concatenate(summary_statistics_list).transpose(0,2,1)
    return {"s_plus_sum": edges_sim[:,:,-2].sum(axis = 1), 
            "s_minus_sum": edges_sim[:,:,-1].sum(axis = 1)}


# def create_s_update_X(X_t, edges_iter, edge_per_t, parameters, rho, mu_plus, mu_minus, 
#                      summary_statistics_list, X_list):
#     edges_t = next(edges_iter, None)
#     if edges_t is not None:
#         epsilon_plus,epsilon_minus, obs_f = epsilons_from_theta(parameters, dict_theta = True, numpy = True)
#         u = edges_t.T[:obs_f,:]
#         v = edges_t.T[-2,:]
        
#         diff_X = X_t[u].mean(axis = 0) - X_t[v]
#         s_plus = ((np.random.rand(edge_per_t) < np_sigmoid(rho * (epsilon_plus - np.abs(diff_X))))) + 0
#         s_minus = ((np.random.rand(edge_per_t) < np_sigmoid(-rho * (epsilon_minus - np.abs(diff_X))))) + 0

#         updates_plus = mu_plus * s_plus * diff_X 
#         updates_minus = mu_minus * s_minus * diff_X 
#         X_t[v] += updates_plus - updates_minus
#         X_t[v] = np.clip(X_t[v], 1e-5, 1 - 1e-5)
#         X_list.append(X_t[None,:].copy())
#         summary_statistics_list.append(np.concatenate([s_plus[None,:], s_minus[None,:]])[None,:])
#         create_s_update_X(X_t, edges_iter, edge_per_t, parameters, rho, mu_plus, mu_minus, summary_statistics_list, X_list)
#     edges_sim = np.concatenate(summary_statistics_list).transpose(0,2,1)
#     return {"s_plus_sum": edges_sim[:,:,-2].sum(axis = 1), 
#             "s_minus_sum": edges_sim[:,:,-1].sum(axis = 1)}


def create_trajectory(X0, edges, parameters, rho, mu_plus, mu_minus):
    X0 = X0.copy()
    edges_iter = (edges_t for edges_t in edges)
    T, edge_per_t, _ = edges.shape
    summary_statistics = create_summary_statistics(X0, edges_iter, edge_per_t, parameters, rho, mu_plus, mu_minus)
    # summary_statistics = create_s_update_X(X0, edges_iter, edge_per_t, parameters, rho, mu_plus, mu_minus, [], [X0[None,:].copy()])
    return summary_statistics

def sim_trajectory_X0_edges(X0, edges, rho, mu_plus, mu_minus):
    return lambda parameters: create_trajectory(X0, edges, parameters, rho, mu_plus, mu_minus)



def train_abc(X, edges, max_f_possible, mu_plus, mu_minus, populations_budget = 10, 
              intermediate_populations = None, population_size = 200, rho = 32, 
              id = None, timeout = 3600):
    if intermediate_populations is None:
        intermediate_populations = populations_budget
    T = len(X)
    res = []
    tot_time = 0
    model_abc = sim_trajectory_X0_edges(X[0], edges, rho, mu_plus, mu_minus)
    prior = pyabc.Distribution(
                theta0=pyabc.RV("norm", 0, 1),
                theta1=pyabc.RV("norm", 0, 1),
                thetaf = pyabc.RV('rv_discrete', values=(np.arange(max_f_possible), np.ones(max_f_possible) / max_f_possible))
                )
    distance = pyabc.PNormDistance(2)
    obs = {"s_plus_sum": edges[:,:,-2].sum(axis = 1), 
           "s_minus_sum": edges[:,:,-1].sum(axis = 1)}
    abc = pyabc.ABCSMC(model_abc, prior, distance, population_size = population_size)#, sampler = SingleCoreSampler())
    db = "sqlite:///" + os.path.join(gettempdir(), f"{id}_feed_test.db")
    history = abc.new(db, obs)
    run_id = history.id
    for _ in range(int(populations_budget / intermediate_populations)):
        abc_continued = pyabc.ABCSMC(model_abc, prior, distance, population_size = population_size)#, sampler = SingleCoreSampler())
        abc_continued.load(db, run_id)
        t0 = time()
        history = abc_continued.run(max_nr_populations = intermediate_populations, minimum_epsilon =  10 * (T ** (1/2)),
                                     max_walltime = timedelta(hours = 3))
        t1 = time()
        tot_time += (t1 - t0)
        theta_samples = jnp.array(history.get_distribution()[0])
        epsilon_mean, epsilon_std, mode_samples_feed,mean_samples_feed = analyse_samples(theta_samples, gumbel_softmax = False)
        res.append({"method": "abc",
                    "id": id,
                    "epsilon_mean": epsilon_mean, 
                    "epsilon_std": epsilon_std, 
                    "argmax_samples_feed": mode_samples_feed,
                    "feed_mean_est": mean_samples_feed,
                    "tot_time": tot_time,
                    "n_simulations": history.total_nr_simulations, 
                    "n_steps": None,
                    "n_samples": None,
                    })

        if tot_time > timeout:
            break

    return res


    


def complete_experiment(N, T, edge_per_t, max_f_possible, rho = 32, obs_f = None,
                        epsilon_plus = None, epsilon_minus = None, mu_plus = 0.02, mu_minus = 0.02,
                        n_steps = 400, n_samples = 40, intermediate_steps = None,  num_chains = 1,
                        intermediate_samples = None, warmup_samples = None, 
                        populations_budget = 10, population_size = 200, id = None, intermediate_populations = None,
                        method = "svinormal", progress_bar = False, date = None, save_data = True, timeout = 25000
                        ):
    

    if len(glob(f"../data/feed_{date}/X_{id}*")) > 0:
        X_file = glob(f"../data/feed_{date}/X_{id}*")[0]
        edges_file = glob(f"../data/feed_{date}/edges_{id}*")[0]
        X = np.load(X_file)
        edges = np.load(edges_file)
        
        _,_,epsilon_plus,epsilon_minus, obs_f = [int(u) for u in X_file.split("/")[-1].split("_")[2:-1]]
        epsilon_plus, epsilon_minus = np.array([epsilon_plus, epsilon_minus]) / 100
    else:
        if epsilon_plus is None:
            epsilon_plus = np.random.randint(5) * 0.1 + 0.05
            epsilon_minus = np.random.randint(5) * 0.1 + 0.55
        if obs_f is None:
            obs_f = np.random.randint(max_f_possible) + 1

        X, edges = BC_feed.simulate_trajectory(N = N, T = T, edge_per_t = edge_per_t, 
                                                max_f_possible = max_f_possible, obs_f = obs_f, epsilon_plus = epsilon_plus, 
                                                epsilon_minus = epsilon_minus, mu_plus = mu_plus, mu_minus = mu_minus, rho = rho)
        if save_data:
            np.save(f"../data/feed_{date}/X_{id}_{int(epsilon_plus * 100)}_{int(epsilon_minus * 100)}_{obs_f}_.npy", X)
            np.save(f"../data/feed_{date}/edges_{id}_{int(epsilon_plus * 100)}_{int(epsilon_minus * 100)}_{obs_f}_.npy", edges)
    
    analysis_data = count_interactions(X, edges, max_f_possible)

    out = []
    
    if method == "svinormal":
        res_svinormal = train_svi(X, edges, max_f_possible = max_f_possible, guide_family = "normal", 
                            n_steps = n_steps, intermediate_steps = intermediate_steps,
                            id = id, rho = rho, timeout = timeout)
        out += res_svinormal                    
    
    if method == "sviNF":
        res_svinf = train_svi(X, edges, max_f_possible = max_f_possible, guide_family = "NF", 
                            n_steps = n_steps, intermediate_steps = intermediate_steps,
                            id = id, rho = rho, timeout = timeout)
        out += res_svinf                    
        
    
    if method == "mcmc":
        res_mcmc = train_mcmc(X, edges, max_f_possible = max_f_possible, n_samples = n_samples, num_chains = num_chains, 
                              intermediate_samples = intermediate_samples, warmup_samples = warmup_samples, 
                              id = id, rho = rho, timeout = timeout)
        out += res_mcmc                      
    
    if method == "abc":
        res_abc = train_abc(X, edges, max_f_possible = max_f_possible, mu_plus = mu_plus, mu_minus = mu_minus, populations_budget = populations_budget, 
              intermediate_populations = intermediate_populations, population_size = population_size, rho = rho, id = id, timeout = timeout)
        out += res_abc

    complete_analysis = [analyse_results(epsilon_plus, epsilon_minus, obs_f, max_f_possible, **res)|analysis_data for res in out]
    return complete_analysis


if __name__ == '__main__':
    T, max_f_possible = [int(sys.argv[k + 2]) for k in range(2)]
    method = sys.argv[4]
    date = sys.argv[5]
    rep = sys.argv[1]
    edge_per_t = 10

    id = f"{rep}_{max_f_possible}_{T}"
    
    if not os.path.exists(f"../data/feed_{date}"):
        try:
            os.mkdir(f"../data/feed_{date}")
        except:
            None
    path= f"../data/feed_{date}/estimation_T{T}_maxf{max_f_possible}_rep{rep}_method{method}.pkl"

    print(f"++++++ feed rep {rep} start {T} {max_f_possible} {method} ++++++")
    
    t0 = time()
    experiment = complete_experiment(N = 400, T = T, edge_per_t = edge_per_t, max_f_possible = max_f_possible, 
                                     method = method, id = id, date = date,
                                     n_steps = 20000, n_samples = 800, populations_budget = 40, 
                                     #intermediate_populations = 5, 
                                     population_size = 5000)
    
    save_pickle(experiment, path)
    print(f">>>>>>>> rep {rep} save {T} {max_f_possible} {method} {round(time() - t0)}s<<<<<<< ")
    

