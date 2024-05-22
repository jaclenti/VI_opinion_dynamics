import sys 
sys.path += ["../src"]
import BC_leaders
import jax, jaxlib
from pyABC_.pyabc.sampler import SingleCoreSampler
import jax.numpy as jnp
import numpy as np
from importlib import reload
import matplotlib.pyplot as plt
from tqdm import tqdm
import jax.random as random
from scipy.special import expit as np_sigmoid
import numpyro
from time import time
from jax.scipy.special import expit as sigmoid
from numpyro.infer import SVI, Trace_ELBO, TraceGraph_ELBO, MCMC, NUTS
from numpyro.infer.autoguide import AutoNormal, AutoBNAFNormal, AutoIAFNormal
from pyro.nn import PyroModule
from numpyro import distributions
from numpyro.optim import Adam
from time import time
from datetime import timedelta
import pickle
from glob import glob
import os
from tempfile import gettempdir

from diptest import dipstat
from scipy.stats import kurtosis, skew

from pyABC_ import pyabc

sys.setrecursionlimit(10000)

# numpyro.set_platform('cpu')

def epsilons_from_theta(theta):
    epsilons = sigmoid(theta) / 2 + jnp.array([0.,0.,.5,.5])
    return epsilons

def count_interactions(X, edges, roles):
    T, N = X.shape
    _,edge_per_t,_ = edges.shape
    
    edges_roles = roles[edges[:,:,1].reshape(edge_per_t * (T - 1),1)]
    edges_signs = edges[:,:,-2:].reshape(edge_per_t * (T - 1), 2)
    pos_interactions_L_plus, pos_interactions_L_minus = [u.item() for u in (edges_roles * edges_signs).sum(axis = 0)]
    pos_interactions_F_plus, pos_interactions_F_minus = [u.item() for u in ((1 - edges_roles) * edges_signs).sum(axis = 0)]
    tot_interactions_L, tot_interactions_F = edges_roles.sum().item(), (1 - edges_roles).sum().item()
    
    return {"pos_interactions_L_plus":pos_interactions_L_plus, "pos_interactions_L_minus":pos_interactions_L_minus, 
            "pos_interactions_F_plus":pos_interactions_F_plus, "pos_interactions_F_minus":pos_interactions_F_minus, 
            "tot_interactions_L":tot_interactions_L, "tot_interactions_F":tot_interactions_F, 
            "T": T, "N": N, "edge_per_t": edge_per_t, 
            "initial_leaders": roles.sum().item(), "initial_leaders_ratio": N / roles.sum().item(),
            "var_X_end": X[-1].var(),
            "skew_X_end": skew(X[-1]),
            "kurtosis_X_end": kurtosis(X[-1]),
            "bimodality_X_end": dipstat(X[-1]),
}

def analyse_results(epsilon_plus, epsilon_minus, roles, round_samples_roles, epsilon_mean,
                     epsilon_std, tot_time, n_steps, n_samples, n_simulations, id, method):
    epsilons = jnp.concatenate([jnp.array(epsilon_plus), jnp.array(epsilon_minus)])

    epsilon_names = ["epsilon_F_plus", "epsilon_L_plus", "epsilon_F_minus", "epsilon_L_minus"]
    out = {
        "id": id,
        "method": method,
        "mse_epsilon": ((epsilons - epsilon_mean)**2).mean().item(), 
        "accuracy_roles": ((round_samples_roles == roles) + 0.).mean().item(),
        "correct_leaders": (((round_samples_roles == 1)&(roles == 1)).sum() / (roles == 1).sum()).item(),
        "correct_followers": (((round_samples_roles == 0)&(roles == 0)).sum() / (roles == 0).sum()).item(),
        "est_leaders": round_samples_roles.sum().item(),
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

def initialize_training(X, edges, roles, rho = 32):
    T, N = X.shape
    _,edge_per_t,_ = edges.shape
    
    u,v,s_plus,s_minus,t = BC_leaders.convert_edges_uvst(edges)
    s_plus = jnp.float32(s_plus)
    s_minus = jnp.float32(s_minus)
    X = jnp.array(X)
    diff_X = X[t,u] - X[t,v]
    
    return {"u": u, "v": v, "s_plus": s_plus, "s_minus": s_minus, "t": t, "diff_X": diff_X, "N": N, "rho": rho}

def model(data):
    diff_X,u,v,s_plus, s_minus,t, rho, N = [data[k] for k in ["diff_X","u","v","s_plus", "s_minus","t", "rho", "N"]]
    dim = N + 4
    dist = distributions.Normal(jnp.zeros(dim), jnp.ones(dim)).to_event(1)
    params = numpyro.sample("theta", dist)
    
    theta = params[:4]
    epsilon_plus_F, epsilon_plus_L, epsilon_minus_F, epsilon_minus_L = sigmoid(theta) / 2 + jnp.array([0.,0.,.5,.5])

        
    param_roles = params[4:]
    roles_sample = numpyro.sample("roles", distributions.RelaxedBernoulli(probs = param_roles, temperature = jnp.array([0.1])).to_event(1))
    roles_v = roles_sample[v]
        
    # diff_X = jnp.array(diff_X)
    s_plus = jnp.array(s_plus)
    s_minus = jnp.array(s_minus)
 
    epsilon_plus_r = (1 - roles_v) * epsilon_plus_F + roles_v * epsilon_plus_L
    epsilon_minus_r = (1 - roles_v) * epsilon_minus_F + roles_v * epsilon_minus_L
    kappas_plus = BC_leaders.kappa_plus_from_epsilon(epsilon_plus_r, diff_X, rho, with_jax = True)
    kappas_minus = BC_leaders.kappa_minus_from_epsilon(epsilon_minus_r, diff_X, rho, with_jax = True)
    kappas_ = jnp.concatenate([kappas_minus, kappas_plus])
    s = jnp.concatenate([s_minus, s_plus])

    with numpyro.plate("data", s.shape[0]):
        numpyro.sample("obs", distributions.Bernoulli(probs = kappas_), obs = s)


def analyse_samples(samples):
    epsilon_samples = epsilons_from_theta(samples["theta"][:,:4])
    epsilon_mean, epsilon_std = epsilon_samples.mean(axis = 0), epsilon_samples.std(axis = 0)
    round_samples_roles = samples["roles"].mean(axis = 0).round()

    if round_samples_roles.mean() > 0.5:
        round_samples_roles = 1 - round_samples_roles
        epsilon_mean = epsilon_mean.at[jnp.arange(4)].set(epsilon_mean[jnp.array([1,0,3,2])])
        epsilon_std = epsilon_std.at[jnp.arange(4)].set(epsilon_std[jnp.array([1,0,3,2])])
        
    return epsilon_mean, epsilon_std, round_samples_roles

def train_svi(X, edges, roles, n_steps, intermediate_steps = None, rho = 32, progress_bar = False, 
              lr = 0.01, guide_family = "normal", timeout = 3600, id = None):
    
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
        
    data = initialize_training(X, edges, roles, rho = rho)
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
        epsilon_mean, epsilon_std, round_samples_roles = analyse_samples(svi_samples)

        res.append({
            "id": id,
            "method": "svi" + guide_family,
            "epsilon_mean": epsilon_mean, 
            "epsilon_std": epsilon_std, 
            "round_samples_roles": round_samples_roles,
            "tot_time": tot_time,
            "n_simulations": None,
            "n_steps": intermediate_steps * (_ + 1),
            "n_samples": None,
            })
        last_state = svi_results.state

        if tot_time > timeout:
            break

    return res


def train_mcmc(X, edges, roles, n_samples, intermediate_samples = None, rho = 32, num_chains = 1,
                warmup_samples = None, progress_bar = False, id = None, timeout = 3600):
    if intermediate_samples is None:
        intermediate_samples = n_samples
    if warmup_samples is None:
        warmup_samples = intermediate_samples

    data = initialize_training(X, edges, roles, rho = rho)

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
        epsilon_mean, epsilon_std, round_samples_roles = analyse_samples(mcmc_samples)

        res.append({
            "id": id,
            "method": "mcmc",
            "epsilon_mean": epsilon_mean, 
            "epsilon_std": epsilon_std, 
            "round_samples_roles": round_samples_roles,
            "tot_time": tot_time,
            "n_simulations": None,
            "n_steps": None,
            "n_samples": intermediate_samples * (_ + 1),
            })
        
        if tot_time > timeout:
            break

    return res

##### pyabc ####

def create_summary_statistics(X0, edges_iter, edge_per_t, parameters, rho, mu_plus, mu_minus):
    summary_statistics_list = []
    Xt = X0.copy()
    N = len(Xt)
    
    while True:
        edges_t = next(edges_iter, None)
        if edges_t is None:
            break
        epsilon_plus_F,epsilon_plus_L,epsilon_minus_F,epsilon_minus_L = epsilons_from_theta(np.array([parameters[f"theta{k}"] for k in range(4)]))
        roles = np.array([parameters[f"theta{k + 4}"] for k in range(N)])
        u,v = edges_t.T[:2,:]
        
        epsilon_plus = (1 - roles[v]) * epsilon_plus_F + roles[v] * (epsilon_plus_L)
        epsilon_minus = (1 - roles[v]) * epsilon_minus_F + roles[v] * (epsilon_minus_L)
        
        diff_X = Xt[u] - Xt[v]
        
        s_plus =  (np.abs(diff_X) < epsilon_plus) + 0
        s_minus = (np.abs(diff_X) > epsilon_minus) + 0

        updates_plus = mu_plus[roles[v]] * s_plus * diff_X 
        updates_minus = mu_minus[roles[v]] * s_minus * diff_X 
        Xt[v] += updates_plus - updates_minus
        Xt[v] = np.clip(Xt[v], 1e-5, 1 - 1e-5)
        summary_statistics_list.append(np.concatenate([s_plus[None,:], s_minus[None,:]])[None,:])
    edges_sim = np.concatenate(summary_statistics_list).transpose(0,2,1)

    return {"s_plus_sum": edges_sim[:,:,-2].sum(axis = 1), 
            "s_minus_sum": edges_sim[:,:,-1].sum(axis = 1)}


# def create_s_update_X(X_t, edges_iter, edge_per_t, parameters, rho, mu_plus, mu_minus, 
#                      summary_statistics_list = []):
#     N = len(X_t)
#     edges_t = next(edges_iter, None)
#     if edges_t is not None:
#         epsilon_plus_F,epsilon_plus_L,epsilon_minus_F,epsilon_minus_L = epsilons_from_theta(np.array([parameters[f"theta{k}"] for k in range(4)]))
#         roles = np.array([parameters[f"theta{k + 4}"] for k in range(N)])
        
#         u,v = edges_t.T[:2,:]
        
#         epsilon_plus = (1 - roles[v]) * epsilon_plus_F + roles[v] * (epsilon_plus_L)
#         epsilon_minus = (1 - roles[v]) * epsilon_minus_F + roles[v] * (epsilon_minus_L)
        
#         diff_X = X_t[u] - X_t[v]
#         s_plus = ((np.random.rand(edge_per_t) < np_sigmoid(rho * (epsilon_plus - np.abs(diff_X))))) + 0
#         s_minus = ((np.random.rand(edge_per_t) < np_sigmoid(-rho * (epsilon_minus - np.abs(diff_X))))) + 0

#         updates_plus = mu_plus[roles[v]] * s_plus * diff_X 
#         updates_minus = mu_minus[roles[v]] * s_minus * diff_X 
#         X_t[v] += updates_plus - updates_minus
#         X_t[v] = np.clip(X_t[v], 1e-5, 1 - 1e-5)
#         # X_list.append(X_t[None,:].copy())
#         summary_statistics_list.append(np.concatenate([s_plus[None,:], s_minus[None,:]])[None,:])
#         create_s_update_X(X_t, edges_iter, edge_per_t, parameters, rho, mu_plus, mu_minus, summary_statistics_list)
#     edges_sim = np.concatenate(summary_statistics_list).transpose(0,2,1)
#     return {"s_plus_sum": edges_sim[:,:,-2].sum(axis = 1), 
#             "s_minus_sum": edges_sim[:,:,-1].sum(axis = 1)}


def create_trajectory(X0, edges, parameters, rho, mu_plus, mu_minus):
    X0 = X0.copy()
    edges_iter = (edges_t for edges_t in edges)
    T, edge_per_t, _ = edges.shape
    summary_statistics = create_summary_statistics(X0, edges_iter, edge_per_t, parameters, rho, mu_plus, mu_minus)
    # summary_statistics = create_s_update_X(X0, edges_iter, edge_per_t, parameters, rho, mu_plus, mu_minus, [])
    return summary_statistics

def sim_trajectory_X0_edges(X0, edges, rho, mu_plus, mu_minus):
    return lambda parameters: create_trajectory(X0, edges, parameters, rho, mu_plus, mu_minus)



def train_abc(X, edges, mu_plus, mu_minus, populations_budget = 10, 
              intermediate_populations = None, population_size = 200, rho = 32, 
              id = None, timeout = 3600):
    if intermediate_populations is None:
        intermediate_populations = populations_budget
    

    res = []
    tot_time = 0
    model_abc = sim_trajectory_X0_edges(X[0], edges, rho, mu_plus, mu_minus)
    T, N = X.shape
    prior_dict = {f"theta{k}":pyabc.RV("norm", 0, 1) for k in range(4)}|{f"theta{k + 4}": pyabc.RV('rv_discrete', values=(np.arange(2), 0.5 * np.ones(2))) for k in range(N)}
    prior = pyabc.Distribution(prior_dict)
    distance = pyabc.PNormDistance(2)
    obs = {"s_plus_sum": edges[:,:,-2].sum(axis = 1), 
           "s_minus_sum": edges[:,:,-1].sum(axis = 1)}
    abc = pyabc.ABCSMC(model_abc, prior, distance, population_size = population_size)#, sampler = SingleCoreSampler())
    db = "sqlite:///" + os.path.join(gettempdir(), f"{id}_leaders_test.db")
    history = abc.new(db, obs)
    run_id = history.id
    for _ in range(int(populations_budget / intermediate_populations)):
        abc_continued = pyabc.ABCSMC(model_abc, prior, distance, population_size = population_size)#, sampler = SingleCoreSampler())
        abc_continued.load(db, run_id)
        t0 = time()
        history = abc_continued.run(max_nr_populations = intermediate_populations, minimum_epsilon = 5 * (T ** (1/2)), max_walltime = timedelta(hours=3))
        t1 = time()
        tot_time += (t1 - t0)
        theta_samples = jnp.array(history.get_distribution()[0][[f"theta{k}" for k in range(N + 4)]])
        epsilon_mean, epsilon_std, round_samples_roles = analyse_samples({"theta": theta_samples[:,:4], "roles": theta_samples[:,4:]})
        res.append({
            "id": id,
            "method": "abc",
            "epsilon_mean": epsilon_mean, 
            "epsilon_std": epsilon_std, 
            "round_samples_roles": round_samples_roles,
            "tot_time": tot_time,
            "n_simulations": history.total_nr_simulations,
            "n_steps": None,
            "n_samples": None,
            })
        
        if tot_time > timeout:
            break

    return res

#######


def complete_experiment(N, T, edge_per_t, initial_leaders, rho = 32, 
                        epsilon_plus = None, epsilon_minus = None, mu_plus = np.array([0.02, 0.02]), mu_minus = np.array([0.02, 0.02]),
                        n_steps = 400, n_samples = 40, intermediate_steps = None, num_chains = 1,
                        intermediate_samples = None, warmup_samples = None, intermediate_populations = None,
                        populations_budget = 10, population_size = 200, id = None, date = None, timeout = 25000,
                        method = "svinormal", progress_bar = False, save_data = True
                        ):
    if len(glob(f"../data/leaders_{date}/X_{id}*")) > 0:
        X_file = glob(f"../data/leaders_{date}/X_{id}*")[0]
        edges_file = glob(f"../data/leaders_{date}/edges_{id}*")[0]
        roles_file = glob(f"../data/leaders_{date}/roles_{id}*")[0]
        X = np.load(X_file)
        edges = np.load(edges_file)
        roles = np.load(roles_file)
        
        _,_,_,epsilon_plus_F, epsilon_plus_L,epsilon_minus_F, epsilon_minus_L = [int(u) for u in X_file.split("/")[-1].split("_")[2:-1]]
        epsilon_plus, epsilon_minus = np.array([epsilon_plus_F, epsilon_plus_L]) / 100, np.array([epsilon_minus_F, epsilon_minus_L]) / 100
    else:
        if epsilon_plus is None:
            l_plus = [[round(i, 3),round(j, 3)] for i in np.arange(0.05, 0.5, 0.1) for j in np.arange(0.05,0.5, 0.1) if i >= j]
            l_minus = [[round(1-i, 3),round(1-j, 3)] for i in np.arange(0.05, 0.5, 0.1) for j in np.arange(0.05,0.5, 0.1) if i >= j]

            epsilon_plus = np.array(l_plus[np.random.choice(15)])
            epsilon_minus = np.array(l_minus[np.random.choice(15)])
            epsilon_plus_F, epsilon_plus_L = epsilon_plus
            epsilon_minus_F, epsilon_minus_L = epsilon_minus

        X, edges, roles = BC_leaders.simulate_trajectory(N = N, T = T, edge_per_t = edge_per_t, 
                                                        initial_leaders = initial_leaders, epsilon_plus = epsilon_plus, 
                                                        epsilon_minus = epsilon_minus, mu_plus = mu_plus, mu_minus = mu_minus, rho = 10000)
        
        
        if save_data:

            np.save(f"../data/leaders_{date}/X_{id}_{int(epsilon_plus_F * 100)}_{int(epsilon_plus_L * 100)}_{int(epsilon_minus_F * 100)}_{int(epsilon_minus_L * 100)}_.npy", X)
            np.save(f"../data/leaders_{date}/edges_{id}_{int(epsilon_plus_F * 100)}_{int(epsilon_plus_L * 100)}_{int(epsilon_minus_F * 100)}_{int(epsilon_minus_L * 100)}_.npy", edges)
            np.save(f"../data/leaders_{date}/roles_{id}_{int(epsilon_plus_F * 100)}_{int(epsilon_plus_L * 100)}_{int(epsilon_minus_F * 100)}_{int(epsilon_minus_L * 100)}_.npy", roles)
    
    
    if epsilon_plus is None:
        l_plus = [[round(i, 3),round(j, 3)] for i in np.arange(0.05, 0.5, 0.1) for j in np.arange(0.05,0.5, 0.1) if i >= j]
        l_minus = [[round(1-i, 3),round(1-j, 3)] for i in np.arange(0.05, 0.5, 0.1) for j in np.arange(0.05,0.5, 0.1) if i >= j]

        epsilon_plus = np.array(l_plus[np.random.choice(10)])
        epsilon_minus = np.array(l_minus[np.random.choice(10)])

    # X, edges, roles = BC_leaders.simulate_trajectory(N = N, T = T, edge_per_t = edge_per_t, 
    #                                                 initial_leaders = initial_leaders, epsilon_plus = epsilon_plus, 
    #                                                 epsilon_minus = epsilon_minus, mu_plus = mu_plus, mu_minus = mu_minus, rho = rho)
    edges = edges.astype(int)
    roles = roles.astype(int)

    analysis_data = count_interactions(X, edges, roles)

    out = []

    if method == "svinormal":
        res_svinormal = train_svi(X, edges, roles, n_steps = n_steps, id = id,
                             intermediate_steps = intermediate_steps, 
                             guide_family = "normal", rho = rho, timeout = timeout)
        out += res_svinormal
    if method == "sviNF":
        res_NF = train_svi(X, edges, roles, n_steps = n_steps, id = id,
                             intermediate_steps = intermediate_steps, 
                             guide_family = "NF", rho = rho, timeout = timeout)
        out += res_NF

    if method == "mcmc":
        res_mcmc = train_mcmc(X, edges, roles, n_samples = n_samples, id = id, num_chains = num_chains,
                              intermediate_samples = intermediate_samples, warmup_samples = warmup_samples, rho = rho, timeout = timeout)
        out += res_mcmc

    if method == "abc":
        res_abc = train_abc(X, edges, mu_plus, mu_minus, populations_budget = populations_budget, 
              intermediate_populations = intermediate_populations, population_size = population_size, rho = rho, timeout = timeout,
              id = id)
        out += res_abc

    complete_analysis = [analyse_results(epsilon_plus, epsilon_minus, roles, **res)|analysis_data for res in out]
    return complete_analysis


if __name__ == '__main__':
    T, N, initial_leaders_ratio = [int(sys.argv[k + 2]) for k in range(3)]
    method = sys.argv[5]
    rep = sys.argv[1]
    date = sys.argv[6]
    
    edge_per_t = 10
    

    id = f"{rep}_{N}_{initial_leaders_ratio}_{T}"
    
    if not os.path.exists(f"../data/leaders_{date}"):
        try:
            os.mkdir(f"../data/leaders_{date}")
        except:
            None
    path = f"../data/leaders_{date}/estimation_T{T}_N{N}_lratio{initial_leaders_ratio}_rep{rep}_method{method}.pkl"
    
        
    print(f"++++++ leaders rep {rep} start T{T} N{N} ilr{initial_leaders_ratio} {method} ++++++ ")

    # if ((N / initial_leaders_ratio) < 1)|((T > 2048) & (method in ["abc", "mcmc"]))|((N > 200) & (method == "sviNF")):
    if ((N / initial_leaders_ratio) < 1):
        print(f"skip {method} N{N} T{T} ilr{initial_leaders_ratio}")
    else:
        t0 = time()
        experiment = complete_experiment(N, T, edge_per_t, int(N / initial_leaders_ratio),  
                                        n_steps = 20000,
                                        n_samples = 800, method = method, #intermediate_populations = 5,
                                        populations_budget = 40, population_size = 5000,
                                        id = id, date = date
                                        )
        
        save_pickle(experiment, path)
        print(f">>>>>>>> rep {rep} save {T} {N} {initial_leaders_ratio} {method} {round(time() - t0)}s<<<<<<<")
