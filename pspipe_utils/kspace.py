"""
Some utility functions for the kspace filter.
"""
from pspy import so_spectra, so_map
import numpy as np

def build_kspace_filter_matrix(lb, ps_sims, n_sims, spectra, return_dict=False):

    """This function compute the kspace filter transfer matrix using
    a bunch of simulations,
    The matrix acts on unfiltered power spectra and return filtered power spectra.
    We will use the inverse of the matrix to remove the biais from the kspace filter.
    the elements that we estimate explicitely are
     "TT_to_TT", "EE_to_EE", "BB_to_BB", "EE_to_BB", "BB_to_EE"
     
     ps_filt_x = \sum_y x_to_y ps_unfilt_y
     
     we then set "TE_to_TE", "ET_to_ET", "TB_to_TB", "BT_to_BT" to sqrt(TT_to_TT * EE_to_EE)
     and "EB_to_EB", "BE_to_BE" to "EE_to_EE"
     
    Parameters
    ----------
    lb : 1d array
        the binned multipoles
    ps_sims: dict
        a dictionnary with all simulated power spectrum, form should be
        ps[[key_a, key_b][spec]
        key_a is "filter" or "nofilter"
        key_b is "standard", "noE", or "noB"
        spec is the spectra list ["TT","TE".....]
    n_sims: integer
        the number of simulations
    spectra: list of str
        the spectra list ["TT","TE".....]
    return dict: boolean
        wether to return a dictionnary with the term of the matrix in
        addition to the matrix
    """

    n_bins = len(lb)
    kspace_matrix = np.zeros((9 * n_bins, 9 * n_bins))
    kspace_dict, std = {}, {}

    for spec1 in spectra:
        for spec2 in spectra:
            kspace_dict[f"{spec1}_to_{spec2}"] = np.zeros(n_bins)
        
    elements = ["TT_to_TT", "EE_to_EE", "BB_to_BB", "EE_to_BB", "BB_to_EE"]
    for el in elements: kspace_dict[el] = []
    for i in range(n_sims):
        kspace_dict["TT_to_TT"] += [ps_sims["filter", "standard"][i]["TT"]/ps_sims["nofilter", "standard"][i]["TT"]]
        
        kspace_dict["EE_to_EE"] += [ps_sims["filter", "noB"][i]["EE"]/ps_sims["nofilter", "noB"][i]["EE"]]
        kspace_dict["BB_to_BB"] += [ps_sims["filter", "noE"][i]["BB"]/ps_sims["nofilter", "noE"][i]["BB"]]
        
        kspace_dict["EE_to_BB"] += [ps_sims["filter", "noB"][i]["BB"]/ps_sims["nofilter", "noB"][i]["EE"]]
        kspace_dict["BB_to_EE"] += [ps_sims["filter", "noE"][i]["EE"]/ps_sims["nofilter", "noE"][i]["BB"]]
 
    for el in elements:
        std[el] = np.std(kspace_dict[el], axis=0)
        kspace_dict[el] = np.mean(kspace_dict[el], axis=0)

    elements = ["TE_to_TE", "ET_to_ET", "TB_to_TB", "BT_to_BT"]
    for el in elements:
        kspace_dict[el] = np.sqrt(kspace_dict["TT_to_TT"] * kspace_dict["EE_to_EE"])
        
    elements = ["EB_to_EB", "BE_to_BE"]
    for el in elements:
        kspace_dict[el] = kspace_dict["EE_to_EE"]

    for i, spec1 in enumerate(spectra):
        for j, spec2 in enumerate(spectra):
            for k in range(n_bins):
                kspace_matrix[k + i * n_bins, k + j * n_bins] = kspace_dict[f"{spec1}_to_{spec2}"][k]

    if return_dict:
        return kspace_dict, std, kspace_matrix
    else:
        return kspace_matrix


def deconvolve_kspace_filter_matrix(lb, ps, kspace_filter_matrix, spectra):

    """This function deconvolve the kspace filter transfer matrix
     
     since
     ps_filt_x = \sum_y M_{x_to_y} ps_unfilt_y
     we have
     ps_unfilt_x = \sum_y (M)**-1_{x_to_y} ps_filt_y
     
     
    Parameters
    ----------
    lb : 1d array
        the binned multipoles
    ps: dict
        a dictionnary with the spectra
        ps[spec]
        spec is the spectra list ["TT","TE".....]
    kspace_filter_matrix: 2d array
        a 9 * n_bins, 9 * n_bins matrix that encode the effect of the kspace filter
    spectra: list of str
        the spectra list ["TT","TE".....]
    """


    n_bins = len(lb)

    inv_kspace_mat = np.linalg.inv(kspace_filter_matrix)
    vec = []
    for f in spectra:
        vec = np.append(vec, ps[f])
    vec = np.dot(inv_kspace_mat, vec)
    ps = so_spectra.vec2spec_dict(n_bins, vec, spectra)

    return lb, ps


def filter_map(map, filter, binary, inv_pixwin=None, weighted_filter=False, tol=1e-4, ref=0.9):

    """Filter the map in Fourier space using a predefined filter. Note that we mutliply the maps by a binary mask before
    doing this operation in order to remove pathological pixels
    We also include an option for removing the pixel window function

    Parameters
    ---------
    map: ``so_map``
        the map to be filtered
    binary:  ``so_map``
        a binary mask removing pathological pixels
    filter: 2d array
        a filter applied in fourier space
    inv_pixwin: 2d array
        the inverse of the pixel window function in fourier space
    weighted_filter: boolean
        wether to use weighted filter a la sigurd
    tol, ref: floats
        only in use in the case of the weighted filter, these arg
        remove crazy pixels value in the weight applied

    """

    if weighted_filter == False:
        if inv_pixwin is not None:
            filter *= inv_pixwin
            
        map = so_map.fourier_convolution(map, filter, binary)

    else:
        map.data *= binary.data
        one_mf = (1 - filter)
        rhs    = enmap.ifft(one_mf * enmap.fft(map.data, normalize=True), normalize=True).real
        div    = enmap.ifft(one_mf * enmap.fft(binary.data, normalize=True), normalize=True).real
        del one_mf
        div    = np.maximum(div, np.percentile(binary.data[::10, ::10], ref * 100) * tol)
        map.data -= rhs / div
        del rhs
        del div
    
        if inv_pixwin_lxly is not None:
            ft = enmap.fft(map.data, normalize=True)
            ft  *= inv_pixwin_lxly
            map.data = enmap.ifft(ft, normalize=True).real

    return map


def get_kspace_filter(template, filter_dict):

    """build the kspace filter according to a dictionnary specifying the filter parameters
    Parameters
    ---------
    template: ``so_map``
        a template of the CAR map we want to filter
    filter_dict: dict
        a dictionnary that contains the filter properties
        e.g filter_dict = {..., "type":"binary_cross","vk_mask":[-90, 90], "hk_mask":[-50, 50], ...}
        
    """

    shape, wcs = template.data.shape, template.data.wcs
    if filter_dict["type"] == "binary_cross":
        filter = so_map_preprocessing.build_std_filter(shape, wcs, vk_mask=filter_dict["vk_mask"], hk_mask=filter_dict["hk_mask"], dtype=np.float64)
    elif filter_dict["type"] == "gauss":
        filter = so_map_preprocessing.build_sigurd_filter(shape, wcs, filter_dict["lbounds"], dtype=np.float64)
    else:
        print("you need to specify a valid filter type")

    return filter
