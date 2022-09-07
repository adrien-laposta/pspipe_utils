from pspy import pspy_utils, so_map, so_spectra, so_window, so_mcm, so_cov, sph_tools
from pspipe_utils import simulation, consistency, best_fits
from pspipe_utils import transfer_function as tf_tools
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib as mpl
from pixell import curvedsky
import numpy as np
import pickle
import time
import sys

# Output dir
output_dir = "result_tf"
pspy_utils.create_directory(output_dir)


###############################
# Simulation input parameters #
###############################
ncomp = 3
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
modes = ["TT", "TE", "ET", "EE"]
type = "Dl"
niter = 0

surveys = ["sv1", "sv2"]
arrays = {"sv1": ["ar1"],
          "sv2": ["ar1"]}
n_splits = {"sv1": 2,
            "sv2": 2}

ra0, ra1, dec0, dec1, res = -30, 30, -10, 10, 3
apo_type_survey = "Rectangle"
template = so_map.car_template(ncomp, ra0, ra1, dec0, dec1, res)
binary = so_map.car_template(1, ra0, ra1, dec0, dec1, res)

binary.data[:] = 0
binary.data[1:-1, 1:-1] = 1

lmax = 1400
lmax_sim = lmax + 1000
bin_size = 40

binning_file = f"{output_dir}/binning.dat"
pspy_utils.create_binning_file(bin_size = bin_size,
                               n_bins = 300, file_name = binning_file)

cosmo_params = {"cosmomc_theta":0.0104085, "logA": 3.044, "ombh2": 0.02237,
                "omch2": 0.1200,  "ns": 0.9649, "tau": 0.0544}

fg_components = {"tt": ["kSZ", "tSZ_and_CIB", "cibp", "dust", "radio"],
                 "te": ["radio", "dust"],
                 "ee": ["radio", "dust"],
                 "bb": ["radio", "dust"],
                 "tb": ["radio", "dust"],
                 "eb": []}

fg_params = {"a_tSZ": 3.30, "a_kSZ": 1.60,"a_p": 6.90, "beta_p": 2.08, "a_c": 4.90,
             "beta_c": 2.20, "a_s": 3.10, "xi": 0.1,  "T_d": 9.60,  "a_gtt": 14.0,
             "a_gte": 0.7, "a_pste": 0,
             "a_gee": 0.27, "a_psee": 0.05,
             "a_gbb": 0.13, "a_psbb": 0.05,
             "a_gtb": 0.36, "a_pstb": 0}

rms_range = {"sv1": (1, 3),
             "sv2": (25, 35)}
fwhm_range = {"sv1": (0.5, 2),
              "sv2": (5, 10)}
tf_pars = {
    "bb": {"mu": 1.5,
           "std": 0.1},
    "cc": {"mu": 0.008,
           "std": 0.002}
          }

freq_list = []
nu_eff, rms_uKarcmin_T, bl = {}, {}, {}
tf_dict = {}
ell_tf = np.arange(lmax_sim + 1)

for sv in surveys:
    rms_min, rms_max = rms_range[sv]
    fwhm_min, fwhm_max = fwhm_range[sv]
    for ar in arrays[sv]:
        if sv == "sv1":
            nu_eff[sv, ar] = np.random.randint(80, high=240)
            bb = np.random.normal(tf_pars["bb"]["mu"],
                                  tf_pars["bb"]["std"])
            cc = np.random.normal(tf_pars["cc"]["mu"],
                                  tf_pars["cc"]["std"])
            tf_dict[sv, ar] = tf_tools.tf_model(ell_tf, 1, bb, np.abs(cc))
            plt.figure()
            plt.plot(ell_tf, tf_dict[sv, ar])
            plt.show()
        else:
            nu_eff[sv, ar] = nu_eff["sv1", ar]
            tf_dict[sv, ar] = np.ones(ell_tf.shape)

        rms_uKarcmin_T[sv, f"{ar}x{ar}"] = np.random.uniform(rms_min, high=rms_max)
        freq_list += [nu_eff[sv, ar]]
        fwhm = np.random.uniform(fwhm_min, high=fwhm_max)
        l_beam, bl[sv, ar] = pspy_utils.beam_from_fwhm(fwhm, lmax_sim)

freq_list = list(dict.fromkeys(freq_list))

for sv in surveys:
    for ar1, ar2 in combinations(arrays[sv], 2):
        r = np.random.uniform(0, high = 0.7)
        rms_uKarcmin_T[sv, f"{ar1}x{ar2}"] = r * np.sqrt(rms_uKarcmin_T[sv, f"{ar1}x{ar1}"] * rms_uKarcmin_T[sv, f"{ar2}x{ar2}"])
        rms_uKarcmin_T[sv, f"{ar2}x{ar1}"] = rms_uKarcmin_T[sv, f"{ar1}x{ar2}"]
###############################
###############################


###########################################
# Prepare the data used in the simulation #
###########################################

# CMB power spectra
l_th, ps_dict = pspy_utils.ps_from_params(cosmo_params, type, lmax_sim)
ps_file_name = f"{output_dir}/cmb.dat"
so_spectra.write_ps(ps_file_name, l_th, ps_dict, type, spectra = spectra)

# Beams
for sv in surveys:
    for ar in arrays[sv]:
        beam_file_name = f"{output_dir}/beam_{sv}_{ar}.dat"
        np.savetxt(beam_file_name, np.transpose([l_beam, bl[sv, ar]]))

# Noise power spectra
for sv in surveys:
    for ar1 in arrays[sv]:
        for ar2 in arrays[sv]:
            l_th, nl_th = pspy_utils.get_nlth_dict(rms_uKarcmin_T[sv, f"{ar1}x{ar2}"],
                                                   type,
                                                   lmax_sim,
                                                   spectra=spectra)
            noise_ps_file_name = f"{output_dir}/mean_{ar1}x{ar2}_{sv}_noise.dat"
            so_spectra.write_ps(noise_ps_file_name, l_th, nl_th, type, spectra = spectra)

# Foreground power spectra
fg_dict = best_fits.get_foreground_dict(l_th, freq_list, fg_components, fg_params, fg_norm = None)
fg = {}
for f1 in freq_list:
    for f2 in freq_list:
        fg[f1, f2] = {}
        for spec in spectra:
            fg[f1, f2][spec] = fg_dict[spec.lower(), "all", f1, f2]
        so_spectra.write_ps(f"{output_dir}/fg_{f1}x{f2}.dat", l_th,
                            fg[f1, f2], type, spectra = spectra)

# Window functions
window = {}
for sv in surveys:
    for ar in arrays[sv]:
        window[sv, ar] = so_window.create_apodization(binary, apo_type=apo_type_survey,
                                                      apo_radius_degree = 1)
        window[sv, ar].plot(file_name = f"{output_dir}/window_{sv}_{ar}")

# Mode coupling matrices
spec_name_list = []
mbb_inv_dict = {}
for id_sv_a, sv_a in enumerate(surveys):
    for id_ar_a, ar_a in enumerate(arrays[sv_a]):
        # we need both the window for T and pol, here we assume they are the same
        window_tuple_a = (window[sv_a, ar_a], window[sv_a, ar_a])
        bl_a = (bl[sv_a, ar_a], bl[sv_a, ar_a])

        for id_sv_b, sv_b in enumerate(surveys):
            for id_ar_b, ar_b in enumerate(arrays[sv_b]):

                if  (id_sv_a == id_sv_b) & (id_ar_a > id_ar_b) : continue
                if  (id_sv_a > id_sv_b) : continue
                # the if here help avoiding redondant computation
                # the mode coupling matrices for sv1 x sv2 are the same as the one for sv2 x sv1
                # identically, within a survey, the mode coupling matrix for ar1 x ar2 =  ar2 x ar1
                window_tuple_b = (window[sv_b, ar_b], window[sv_b, ar_b])
                bl_b = (bl[sv_b, ar_b], bl[sv_b, ar_b])

                mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0and2(window_tuple_a,
                                                            win2 = window_tuple_b,
                                                            bl1 = bl_a,
                                                            bl2 = bl_b,
                                                            binning_file = binning_file,
                                                            lmax=lmax,
                                                            type="Dl",
                                                            niter=niter)

                mbb_inv_dict[sv_a, ar_a, sv_b, ar_b] = mbb_inv
                spec_name = f"{sv_a}&{ar_a}x{sv_b}&{ar_b}"
                spec_name_list += [spec_name]
###########################################
###########################################

############################
# Generate the simulations #
############################
f_name_cmb = output_dir + "/cmb.dat"
f_name_noise = output_dir + "/mean_{}x{}_{}_noise.dat"
f_name_fg = output_dir + "/fg_{}x{}.dat"

ps_mat = simulation.cmb_matrix_from_file(f_name_cmb, lmax_sim, spectra)
l, fg_mat = simulation.foreground_matrix_from_files(f_name_fg, freq_list, lmax_sim, spectra)
noise_mat = {}
for sv in surveys:
    l, noise_mat[sv] = simulation.noise_matrix_from_files(f_name_noise,
                                                          sv,
                                                          arrays[sv],
                                                          lmax_sim,
                                                          n_splits[sv],
                                                          spectra)

print("==============")
print("= SIMULATION =")
print("==============\n")

n_sims = 15
ps_all = {}
for iii in range(n_sims):
    t = time.time()
    alms_cmb = curvedsky.rand_alm(ps_mat, lmax=lmax_sim, dtype="complex64")
    fglms = simulation.generate_fg_alms(fg_mat, freq_list, lmax_sim)
    sim_alm = {}

    for sv in surveys:
        signal_alms = {}
        for ar in arrays[sv]:
            signal_alms[ar] = alms_cmb + fglms[nu_eff[sv, ar]]
            for i in range(3):
                signal_alms[ar][i] = curvedsky.almxfl(signal_alms[ar][i], bl[sv, ar])
            # Apply the transfer function
            signal_alms[ar][0] = curvedsky.almxfl(signal_alms[ar][0], tf_dict[sv, ar])

        for k in range(n_splits[sv]):
            noise_alms = simulation.generate_noise_alms(noise_mat[sv], arrays[sv], lmax_sim)
            for ar in arrays[sv]:
                split = sph_tools.alm2map(signal_alms[ar] + noise_alms[ar], template)

                sim_alm[sv, ar, k] = sph_tools.get_alms(split, (window[sv, ar], window[sv, ar]), niter, lmax)

    for id_sv1, sv1 in enumerate(surveys):
        for id_ar1, ar1 in enumerate(arrays[sv1]):
            for id_sv2, sv2 in enumerate(surveys):
                for id_ar2, ar2 in enumerate(arrays[sv2]):

                    if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                    if  (id_sv1 > id_sv2) : continue

                    ps_dict = {}
                    for spec in spectra:
                        ps_dict[spec] = []

                    for s1 in range(n_splits[sv1]):
                        for s2 in range(n_splits[sv2]):
                            if (sv1 == sv2) & (ar1 == ar2) & (s1>s2) : continue

                            l, ps_master = so_spectra.get_spectra_pixell(sim_alm[sv1, ar1, s1],
                                                                         sim_alm[sv2, ar2, s2],
                                                                         spectra=spectra)

                            lb, ps = so_spectra.bin_spectra(l,
                                                            ps_master,
                                                            binning_file,
                                                            lmax,
                                                            type=type,
                                                            mbb_inv=mbb_inv_dict[sv1, ar1, sv2, ar2],
                                                            spectra=spectra)

                            for count, spec in enumerate(spectra):
                                if (s1 == s2) & (sv1 == sv2): continue #discard the auto
                                else: ps_dict[spec] += [ps[spec]]

                    ps_dict_cross_mean = {}
                    for spec in spectra:
                        ps_dict_cross_mean[spec] = np.mean(ps_dict[spec], axis=0)
                        if ar1 == ar2 and sv1 == sv2:
                            # Average TE / ET so that for same array same season TE = ET
                            ps_dict_cross_mean[spec] = (np.mean(ps_dict[spec], axis=0) + np.mean(ps_dict[spec[::-1]], axis=0)) / 2.
                    spec_name = f"{sv1}&{ar1}x{sv2}&{ar2}"
                    if iii == 0:
                        ps_all[spec_name] = []

                    ps_all[spec_name] += [ps_dict_cross_mean]

    print("sim %05d/%05d took %.02f s to compute" % (iii, n_sims, time.time() - t))
############################
############################

#################################
# Compute the transfer function #
#################################
ps_order = [("sv1&ar1", "sv1&ar1"),
            ("sv1&ar1", "sv2&ar1"),
            ("sv2&ar1", "sv2&ar1")]

n_bins = len(lb)
# Get analytical covmat
l_cmb, cmb_dict = best_fits.cmb_dict_from_file(f_name_cmb, lmax, spectra)
l_fg, fg_dict = best_fits.fg_dict_from_files(f_name_fg, freq_list, lmax, spectra)
l_noise, nl_dict = best_fits.noise_dict_from_files(f_name_noise,  surveys, arrays, lmax, spectra, n_splits = n_splits)
f_name_beam = output_dir + "/beam_{}_{}.dat"
l_beam, bl_dict = best_fits.beam_dict_from_files(f_name_beam, surveys, arrays, lmax)


l_cmb, ps_all_th, nl_all_th = best_fits.get_all_best_fit(spec_name_list,
                                                         l_cmb,
                                                         cmb_dict,
                                                         fg_dict,
                                                         nu_eff,
                                                         spectra,
                                                         nl_dict=nl_dict,
                                                         bl_dict=bl_dict)

an_cov_dict = {}
for i, ps1 in enumerate(spec_name_list):
    for j, ps2 in enumerate(spec_name_list):
        if j < i: continue

        na, nb = ps1.split("x")
        nc, nd = ps2.split("x")

        sv_a, ar_a = na.split("&")
        sv_b, ar_b = nb.split("&")
        sv_c, ar_c = nc.split("&")
        sv_d, ar_d = nd.split("&")

        win = {}
        win["Ta"] = window[sv_a, ar_a]
        win["Pa"] = window[sv_a, ar_a]

        win["Tb"] = window[sv_b, ar_b]
        win["Pb"] = window[sv_b, ar_b]

        win["Tc"] = window[sv_c, ar_c]
        win["Pc"] = window[sv_c, ar_c]

        win["Td"] = window[sv_d, ar_d]
        win["Pd"] = window[sv_d, ar_d]

        coupling = so_cov.cov_coupling_spin0and2_simple(win, lmax, niter = niter)

        mbb_inv_ab = mbb_inv_dict[sv_a, ar_a, sv_b, ar_b]
        mbb_inv_cd = mbb_inv_dict[sv_c, ar_c, sv_d, ar_d]

        # Correct for the TF ~ test
        ps_all_th_corr = ps_all_th.copy()
        for n1, n2, spec in ps_all_th:
            sv1, ar1 = n1.split("&")
            sv2, ar2 = n2.split("&")
            if spec[0] == "T":
                ps_all_th_corr[n1, n2, spec] *= tf_dict[sv1, ar1][2:lmax]
            if spec[1] == "T":
                ps_all_th_corr[n1, n2, spec] *= tf_dict[sv2, ar2][2:lmax]
        #print(ps_all_th.keys())
        ###
        analytic_cov = so_cov.generalized_cov_spin0and2(coupling,
                                                        [na, nb, nc, nd],
                                                        n_splits,
                                                        ps_all_th_corr,
                                                        nl_all_th,
                                                        lmax,
                                                        binning_file,
                                                        mbb_inv_ab,
                                                        mbb_inv_cd,
                                                        binned_mcm=True)
        analytic_cov_select = so_cov.selectblock(analytic_cov, modes,
                                                 n_bins, block="TTTT")

        # Correct for the TF
        tf = np.ones(n_bins)
        for n in [na, nb, nc, nd]:
            sv, ar = n.split("&")
            _, tf_sv_ar = pspy_utils.naive_binning(ell_tf, tf_dict[sv, ar], binning_file, lmax)
            tf *= tf_sv_ar

        #an_cov_dict[(na, nb), (nc, nd)] = analytic_cov_select * np.outer(np.sqrt(tf), np.sqrt(tf))
        an_cov_dict[(na, nb), (nc, nd)] = analytic_cov_select


#### TEST analytic covariances ####
for my_spec in spec_name_list:
    mean_a, _, mc_cov = so_cov.mc_cov_from_spectra_list(ps_all[my_spec],
                                                    ps_all[my_spec],
                                                    spectra=spectra)
    na, nb = my_spec.split("x")
    mc_cov = so_cov.selectblock(mc_cov, spectra, n_bins, block = "TTTT")
    an_cov = an_cov_dict[(na, nb), (na, nb)]

    plt.figure()
    plt.plot(lb, an_cov.diagonal(), label = "AN")
    plt.plot(lb, mc_cov.diagonal(), label = "MC")
    plt.legend()
    plt.tight_layout()
    plt.title(my_spec)
    plt.show()
###################################

##############
# We want to estimate the transfer function using a ratio of
# two power spectra psA / psB
ps_name = {"C1": {"A": ("sv1&ar1", "sv1&ar1"),
                  "B": ("sv1&ar1", "sv2&ar1")},
           "C2": {"A": ("sv1&ar1", "sv2&ar1"),
                  "B": ("sv2&ar1", "sv2&ar1")}}

tf_dict_out = {"C1": {"lb": [], "tf": [], "tf_std": [], "tf_biased": []},
               "C2": {"lb": [], "tf": [], "tf_std": [], "tf_biased": []}}

for iii in range(n_sims):

    for comb in ps_name:

        nameA_1, nameA_2 = ps_name[comb]["A"]
        nameB_1, nameB_2 = ps_name[comb]["B"]

        psA = ps_all[f"{nameA_1}x{nameA_2}"][iii]["TT"]
        psB = ps_all[f"{nameB_1}x{nameB_2}"][iii]["TT"]

        ### test ###
        psA_th = ps_all_th[nameA_1, nameA_2, "TT"]
        psB_th = ps_all_th[nameB_1, nameB_2, "TT"]
        psA_th = pspy_utils.naive_binning(l_cmb, psA_th, binning_file, lmax)[1]
        psB_th = pspy_utils.naive_binning(l_cmb, psB_th, binning_file, lmax)[1]
        ############

        covAA = an_cov_dict[(nameA_1, nameA_2), (nameA_1, nameA_2)]
        covAB = an_cov_dict[(nameA_1, nameA_2), (nameB_1, nameB_2)]
        covBB = an_cov_dict[(nameB_1, nameB_2), (nameB_1, nameB_2)]

        # Select high SNR multipoles
        snr = pspy_utils.naive_binning(l_cmb, cmb_dict["TT"],
                                       binning_file, lmax)[1] / np.sqrt(covBB.diagonal())

        id = np.where(snr >= 4)[0]
        print(f"Old {len(psA)}")
        psA, psB = psA[id], psB[id]
        print(f"New {len(psA)}")
        covAA, covAB, covBB = covAA[np.ix_(id, id)], covAB[np.ix_(id, id)], covBB[np.ix_(id, id)]
        lb_tf = lb[id]
        tf_biased = psA / psB
        tf = tf_tools.get_tf_unbiased_estimator(psA, psB, covAB, covBB)
        tf_cov = tf_tools.get_tf_estimator_covariance(psA_th[id], psB_th[id], covAA, covAB, covBB)

        tf_dict_out[comb]["tf"].append(tf)
        tf_dict_out[comb]["tf_biased"].append(tf_biased)
        tf_dict_out[comb]["tf_std"].append(np.sqrt(tf_cov.diagonal()))
        tf_dict_out[comb]["lb"].append(lb_tf)

# Compute MC errorbars on the TF
for comb in ps_name:

    mean_tf = np.mean(tf_dict_out[comb]["tf"], axis = 0)
    mean_analytical_tf_std = np.mean(tf_dict_out[comb]["tf_std"], axis = 0)

    N = len(mean_tf)
    mc_cov_tf = np.zeros((N, N))
    for iii in range(n_sims):
        mc_tf = tf_dict_out[comb]["tf"][iii]
        mc_cov_tf += np.outer(mc_tf, mc_tf)
    mc_cov_tf /= n_sims
    mc_cov_tf -= np.outer(mean_tf, mean_tf)

    mc_tf_std = np.sqrt(mc_cov_tf.diagonal())


    plt.figure(figsize = (8, 6))
    plt.plot(tf_dict_out[comb]["lb"][0], mean_analytical_tf_std, label = "AN")
    plt.plot(tf_dict_out[comb]["lb"][0], mc_tf_std, label = "MC")
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$\sigma_{F_\ell}$")
    plt.legend(frameon = False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/std_tf.png", dpi = 300)

    mean_tf_biased = np.mean(tf_dict_out[comb]["tf_biased"], axis = 0)

    plt.figure(figsize = (8, 6))
    grid = plt.GridSpec(4, 1, hspace = 0, wspace = 0)
    upper = plt.subplot(grid[:3], xticklabels = [], ylabel = r"$F_\ell^T$")


    upper.plot(ell_tf, tf_dict["sv1", "ar1"], color = "k", ls = "--", label = "Model")
    upper.errorbar(tf_dict_out[comb]["lb"][0], mean_tf, yerr = mc_tf_std,
                 marker = ".", capsize = 0.7, ls = "None", label = "Estimated TF")
    upper.plot(tf_dict_out[comb]["lb"][0], mean_tf_biased, label = "Biased TF")
    upper.fill_between(tf_dict_out[comb]["lb"][0], mean_tf - mean_analytical_tf_std,
                       mean_tf + mean_analytical_tf_std, color = "gray", alpha = 0.2)
    upper.legend(frameon = False)
    upper.set_xlim(right = lmax)

    xlims = upper.get_xlim()
    lower = plt.subplot(grid[-1], xlabel = r"$\ell$", ylabel = r"$\Delta F_\ell^T$",
                        xlim = xlims)

    lower.axhline(0, color = "k", ls = "--")
    _, tf_sv_ar = pspy_utils.naive_binning(ell_tf, tf_dict["sv1", "ar1"], binning_file, lmax)
    lower.errorbar(tf_dict_out[comb]["lb"][0], mean_tf - tf_sv_ar[id], yerr = mc_tf_std,
                   marker = ".", capsize = 0.7, ls = "None", label = "Estimated TF")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/estimation_tf.png", dpi = 300)
