from scipy.optimize import linear_sum_assignment
from arnie_utils import *

def _hungarian(bpp, exp=1, sigmoid_slope_factor=None, prob_to_0_threshold_prior=0,
               prob_to_1_threshold_prior=1, theta=0, ln=False, add_p_unpaired=True,
               allowed_buldge_len=0, min_len_helix=2):

    bpp_orig = bpp.copy()

    if add_p_unpaired:
        p_unpaired = 1 - np.sum(bpp, axis=0)
        for i, punp in enumerate(p_unpaired):
            bpp[i, i] = punp

    # apply prob_to_0 threshold and prob_to_1 threshold
    bpp = np.where(bpp < prob_to_0_threshold_prior, 0, bpp)
    bpp = np.where(bpp > prob_to_1_threshold_prior, 1, bpp)

    # aply exponential. On second thought this is likely not as helpful as sigmoid since
    # * for 0 < exp < 1 lower probs will increase more than higher ones (seems undesirable)
    # * for exp > 1 all probs will decrease, which seems undesirable (but at least lower probs decrease more than higher ones)
    bpp = np.power(bpp, exp)

    # apply log which follows botlzamann where -ln(P) porportional to Energy
    if ln:
        bpp = np.log(bpp)

    bpp = np.where(np.isneginf(bpp), -1e10, bpp)
    bpp = np.where(np.isposinf(bpp), 1e10, bpp)

    # apply sigmoid modified by slope factor
    if sigmoid_slope_factor is not None and np.any(bpp):
        bpp = _sigmoid(bpp, slope_factor=sigmoid_slope_factor)

        # should think about order of above functions and possibly normalize again here

        # run hungarian algorithm to find base pairs
    _, row_pairs = linear_sum_assignment(-bpp)
    bp_list = []
    for col, row in enumerate(row_pairs):
        # if bpp_orig[col, row] != bpp[col, row]:
        #    print(col, row, bpp_orig[col, row], bpp[col, row])
        if bpp_orig[col, row] > theta and col < row:
            bp_list.append([col, row])

    structure = convert_bp_list_to_dotbracket(bp_list, bpp.shape[0])
    structure = post_process_struct(structure, allowed_buldge_len, min_len_helix)
    bp_list = convert_dotbracket_to_bp_list(structure, allow_pseudoknots=True)

    return structure, bp_list