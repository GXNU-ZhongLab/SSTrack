import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist
# from lib.test.analysis.evaluate_vos import evaluate_vos


trackers = []
dataset_name = 'lasot'  # lasot_extension_subset

trackers.extend(trackerlist(name='sstrack', parameter_name='dropmae_256_150ep', dataset_name=dataset_name,
                            run_ids=150, display_name='sstrack'))

# For VOT evaluate
dataset = get_dataset(dataset_name)
# dataset = get_dataset('otb', 'nfs', 'uav', 'tc128ce')
# plot_results(trackers, dataset, 'OTB2015', merge_results=True, plot_types=('success', 'norm_prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))

# print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))


# For VOS evaluate
# evaluate_vos(trackers, dataset='dv2017_val', force=False) # yt2018_jjval, dv2017_val
