from getdist.mcsamples import loadMCSamples
import matplotlib.pyplot as plt
from pspy import pspy_utils
from cobaya.run import run
import numpy as np

def get_tf_estimator_bias(psA, psB, covAB, covBB):

    alpha = covBB / np.outer(psB, psB)
    alpha -= covAB / np.outer(psA, psB)

    return alpha.diagonal()

def get_tf_unbiased_estimator(psA, psB, covAB, covBB):

    bias = get_tf_estimator_bias(psA, psB, covAB, covBB)

    tf = psA / psB * (1 - bias)

    return tf

def get_tf_estimator_covariance(psA, psB, covAA, covAB, covBB):

    bias = get_tf_estimator_bias(psA, psB, covAB, covBB)
    tf = get_tf_unbiased_estimator(psA, psB, covAB, covBB)

    cov = covAA / np.outer(psA, psA)
    cov += covBB / np.outer(psB, psB)
    cov -= 2 * covAB / np.outer(psA, psB)
    cov -= np.outer(bias, bias)

    cov *= np.outer(tf, tf)

    return cov

def tf_model(ell, aa, bb, cc, method = "logistic"):

    if method == "sigurd":
        x = 1 / (1 + (ell / bb) ** aa)
        tf = x / (x + cc)

    if method == "thib":
        tf = aa + (1 - aa) * np.sin(np.pi / 2 * (ell - bb) / (cc - bb)) ** 2
        tf[ell > cc] = 1

    if method == "beta":
        tf = np.zeros(len(lb))
        id = np.where(ell < bb)
        tf[id] = aa + (1-aa) / (1 + (ell[id] / (bb - ell[id])) ** (-cc))
        tf[ell >= bb] = 1

    if method == "logistic":
        tf = aa / (1 + bb * np.exp(-cc * ell))

    return tf

def fit_tf(lb, tf_est, tf_cov, prior_dict, chain_name, method = "logistic"):

    def loglike(aa, bb, cc):
        res = tf_est - tf_model(lb, aa, bb, cc, method = method)
        chi2 = res @ np.linalg.inv(tf_cov) @ res
        return -0.5 * chi2

    info = {"likelihood": {
                "my_like": loglike},
            "params": {
                "aa": {
                    "prior": prior_dict["aa", method],
                    "latex": "aa"},
                "bb": {
                    "prior": prior_dict["bb", method],
                    "latex": "bb"},
                "cc": {
                    "prior": prior_dict["cc", method],
                    "latex": "cc"},},
            "sampler": {
                "mcmc": {
                    "max_tries": 1e8,
                    "Rminus1_stop": 0.005,
                    "Rminus1_cl_stop": 0.05}},
            "output": chain_name,
            "force": True
            }
    updated_info, sampler = run(info)

def get_parameter_mean_and_std(chain_name, pars):

    s = loadMCSamples(chain_name, settings = {"ignore_rows": 0.5})

    mean = s.mean(pars)
    cov = s.cov(pars)

    return mean, np.sqrt(cov.diagonal())

def get_tf_bestfit(ell, chain_name, method = "logistic"):

    mu, _ = get_parameter_mean_and_std(chain_name, ["aa", "bb", "cc"])
    aa, bb, cc = mu

    tf = tf_model(ell, aa, bb, cc, method = method)

    return tf

def plot_tf(lb_list, tf_list, tf_err_list, titles, plot_file, ell = None, tf_model_list = None):

    n = len(tf_list)
    fig, axes = plt.subplots(2, 3, sharey = True, figsize = (16, 9))

    for i in range(n):
        ax = axes[i//3, i%3]
        if i % 3 == 0:
            ax.set_ylabel(r"$F_\ell^T$")
        if i // 3 == 1:
            ax.set_xlabel(r"$\ell$")

        ax.axhline(1, color = "k", ls = "--", lw = 0.8)
        if tf_model_list is not None:
            ax.plot(ell, tf_model_list[i], color = "gray")
        ax.errorbar(lb_list[i], tf_list[i], yerr = tf_err_list[i], marker = ".",
                    capsize = 1, elinewidth = 1.1, ls = "None", color = "tab:red")
        ax.set_title(titles[i])
        ax.set_ylim(0, 1.3)

    plt.tight_layout()
    plt.savefig(plot_file, dpi = 300)

def downgrade_binning(ps, cov, downgrade, binning_file, lmax):

    bmin, bmax, _, _ = read_binning_file(binning_file, lmax)
    n_bins_in = len(ps)
    n_bins_out = n_bins_in // downgrade
    n_res = n_bins_in % downgrade

    ps_out = ps[:n_bins_out*downgrade].reshape(n_bins_out, downgrade).mean(axis = 1)
    if n_res != 0:
        ps_out = np.concatenate((ps_out, np.array([np.mean(ps[n_bins_out*downgrade:])])))

    cov_out = cov[:n_bins_out*downgrade,
                  :n_bins_out*downgrade].reshape(n_bins_out, downgrade, n_bins_out, downgrade).mean(axis = (1, -1))
    if n_res != 0:
        cov12 = cov[:n_bins_out*downgrade, n_bins_out*downgrade:]

        cov12 = cov12.reshape(n_bins_out, downgrade, 1, n_res).mean(axis=(1,-1))

        cov22 = np.array([[np.mean(cov[-n_bins_in:, -n_bins_in:])]])
        cov_out = [[cov_out, cov12],
                   [cov12.T, cov22]]

        cov_out = np.block(cov_out)


    bmin_out = bmin[::downgrade]

    if n_res != 0:
        bmax_out = np.concatenate((bmax[downgrade - 1::downgrade], [bmax[-1]]))
    else:
        bmax_out = bmax[downgrade - 1::downgrade]
    lb_out = (bmax_out + bmin_out) / 2


    return lb_out, ps_out, cov_out
