import os
import pickle
from sample_ordinal import plot_and_save as plot_and_save_ordinal_results
from sample_sparse_bayes_linear import plot_and_save as plot_and_save_sbl_results
from plotting_scripts.helper_fns import add_results2_to_results1

'''When we run scripts like `sample_ordinal' or `sample_sparse_bayes_linear', error metric plots are created like those
in Figure 2 & Figure 4 of the paper. The data used to create those plots is automatically saved, and can be reloaded
and replotted as shown below. This is helpful for fast re-plotting if we decide to tweak the plotting code
'''

problem_type = "ordinal"
# problem_type = "sbl"

# define the path where data is saved
if problem_type == "ordinal":
    path = f"C:/Users/benja/Code/ebm/results/ordinal/dim20/mixture50_poly2_ssize50/final_10min/results.pkl"
    # path = f"C:/Users/benja/Code/ebm/results/ordinal/dim20/mixture50_poly4_ssize50/final_10min/results.pkl"
elif problem_type == "sbl":
    path = "C:/Users/benja/Code/ebm/results/sbl/20_100/final_10min/results.pkl"
else:
    raise ValueError

# loaded a dictionary containing all the data needed to recreate the plot
with open(path, "rb") as f:
    res = pickle.load(f)

# replot this data
if problem_type == "ordinal":
    plot_and_save_ordinal_results(**res)
elif problem_type == "sbl":
    plot_and_save_sbl_results(**res)


######## Example of loading two different sets of results and combining them #############
# path = "C:/Users/benja/Code/ebm/results/ordinal/dim20/mixture50_poly4_ssize50/old_final_10min/results.pkl"
# with open(os.path.join(path), "rb") as f:
#     res1 = pickle.load(f)
# path = "C:/Users/benja/Code/ebm/results/ordinal/dim20/mixture50_poly4_ssize50/pavg_final_10min/results.pkl"
# with open(os.path.join(path), "rb") as f:
#     res2 = pickle.load(f)
#
# save_dir = "C:/Users/benja/Code/ebm/results/ordinal/dim20/mixture50_poly4_ssize50/final_10min"
# add_results2_to_results1(res1, res2)
# os.makedirs(save_dir, exist_ok=True)
# res1["save_dir"] = save_dir
# plot_and_save(**res1)
