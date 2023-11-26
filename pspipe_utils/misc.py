"""
Some general utility functions that do not belong to other files.
"""

from pspy import pspy_utils
from pixell import curvedsky

def str_replace(my_str, old, new):
    """
    just like replace but check that the replacement actually happened

    Parameters
    __________
    my_str: string
        the string in which the replacment will happen
    old: string
        old part of the string to be replaced
    new: string
        what will replace old
    """

    my_new_str = my_str.replace(old, new)
    if my_new_str == my_str:
        error = f" the name '{my_str}' does not contain '{old}' so I can't replace '{old}' by '{new}'"
        raise NameError(error)
    return my_new_str

def read_beams(f_name_beam_T, f_name_beam_pol, lmax=None):
    """
    read T and pol beams and return a beam dictionnary with entry T, E, B

    Parameters
    __________
    f_name_beam_T: string
        the filename of the temperature beam file
    f_name_beam_pol: string
        the filename of the polarisation beam file
    lmax : integer
        the maximum multipole to consider (note that usually beam file start at l=0)
    """

    bl = {}
    l, bl["T"] = pspy_utils.read_beam_file(f_name_beam_T, lmax=lmax)
    l, bl["E"] = pspy_utils.read_beam_file(f_name_beam_pol, lmax=lmax)
    bl["B"] = bl["E"]
    return l, bl

def apply_beams(alms, bl):
    """
    apply T and pol beams to alms

    Parameters
    __________
    alms: 2d array
        array of alms, alms[0]=alm_T, alms[1]=alm_E, alms[2]=alm_B
    bl: dict
        dictionnary containing T and E,B beams
    """
    for i, f in enumerate(["T", "E", "B"]):
        alms[i] = curvedsky.almxfl(alms[i], bl[f])
    return alms

def get_split_beam_fnames(f_name_beam_T, f_name_beam_pol, survey, id_split):
    """
    Handles the different naming conventions of DR6 and Planck
    per-split beams.

    Parameters
    __________
    f_name_beam_T: string
        the filename of the temperature beam file (coadd)
    f_name_beam_pol: string
        the filename of the polarisation beam file (coadd)
    survey: string
        the name of the survey -should be either "Planck" or "dr6"
    id_split: integer
        the id of the split

    Returns
    _______
    split_beam_T_fname: string
        the filename of the temperature beam file (split)
    split_beam_pol_fname: string
        the filename of the polarisation beam file (split)
    """
    if survey == "Planck":
        split_name = ["A", "B"][id_split]
        split_beam_T_fname = str_replace(
            f_name_beam_T, "_mean.dat", f"{split_name}.dat"
        )
        split_beam_pol_fname = str_replace(
            f_name_beam_pol, "_mean.dat", f"{split_name}.dat"
        )
    elif survey == "dr6":
        split_beam_T_fname = str_replace(
            f_name_beam_T, "coadd", f"set{id_split}"
        )
        split_beam_pol_fname = str_replace(
            f_name_beam_pol, "coadd", f"set{id_split}"
        )
    return split_beam_T_fname, split_beam_pol_fname

