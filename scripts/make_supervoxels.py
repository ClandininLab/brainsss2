import os
import sys
import numpy as np
import h5py
import time
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph
from brainsss2.argparse_utils import get_base_parser  # noqa
from brainsss2.logging_utils import setup_logging  # noqa
import logging
import nibabel as nib
import shutil
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn
"""
Suppressing this warning from AgglomerativeClustering:
UserWarning: Persisting input arguments took 1.06s to run.
If this happens often in your code, it can cause performance problems
(results will be correct in all cases).
The reason for this is probably some large input arguments for a wrapped
 function (e.g. large strings).
THIS IS A JOBLIB ISSUE. If you can, kindly provide the joblib's team with an
 example so that they can fix the problem.
  **kwargs)
 """


def parse_args(input, allow_unknown=True):
    parser = get_base_parser("anatomical_registration")
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="overwrite existing transforms dir",
    )
    parser.add_argument(
        "--nclusters",
        type=int,
        default=2000,
        help="number of clusters to use for supervoxelization",
    )
    parser.add_argument(
        "-d", "--dir", type=str, help="func dir to be processed", required=True
    )
    parser.add_argument(
        "--funcfile",
        type=str,
        default="regression/model000_confound/residuals.nii",
        help="functional file to use",
    )
    parser.add_argument(
        "--linkage",
        type=str,
        default="ward",
        help="linkage method to use for clustering",
    )
    if allow_unknown:
        args, unknown = parser.parse_known_args()
        if unknown is not None:
            print(f"skipping unknown arguments:{unknown}")
    else:
        args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args(sys.argv[1:])

    args = setup_logging(args, logtype="supervoxels")

    ### LOAD BRAIN ###

    logging.info(f"loading brain from {args.funcfile}")
    if args.funcfile.endswith(".nii"):
        brain = nib.load(os.path.join(args.dir, args.funcfile))
    elif args.funcfile.endswith(".h5"):
        f = h5py.File(os.path.join(args.dir, args.funcfile), "r")
        brain = f["data"]
    else:
        raise ValueError(f"Unknown file type: {args.funcfile}")

    ### MAKE CLUSTER DIRECTORY ###

    cluster_dir = os.path.join(args.dir, "clustering")
    if os.path.exists(cluster_dir) and args.overwrite:
        logging.info("overwriting existing cluster dir")
        shutil.rmtree(cluster_dir)
    elif os.path.exists(cluster_dir):
        raise FileExistsError(
            f"cluster dir {cluster_dir} already exists, use -o to overwrite"
        )
    os.mkdir(cluster_dir)

    logging.info("fitting clusters")
    t0 = time.time()
    connectivity = grid_to_graph(brain.shape[0], brain.shape[1])
    cluster_labels = []
    cluster_img = nib.Nifti1Image(
        np.zeros(brain.shape[:-1]), brain.affine, brain.header
    )
    for z in range(brain.shape[2]):
        if args.verbose:
            logging.info(f"clustering slice {z}")
        if isinstance(brain, nib.nifti1.Nifti1Image):
            neural_activity = brain.dataobj[:, :, z, :].reshape(-1, 3384)
        else:
            neural_activity = brain[:, :, z, :].reshape(-1, 3384)
        cluster_model = AgglomerativeClustering(
            n_clusters=args.nclusters,
            memory=cluster_dir,
            linkage=args.linkage,
            connectivity=connectivity,
        )
        cluster_model.fit(neural_activity)
        cluster_labels.append(cluster_model.labels_)
        cluster_img.dataobj[:, :, z] = cluster_model.labels_.reshape(brain.shape[:2])
    cluster_labels = np.asarray(cluster_labels)

    logging.info("saving clustering solution")
    save_file = os.path.join(cluster_dir, "cluster_labels.npy")
    np.save(save_file, cluster_labels)
    save_img = os.path.join(cluster_dir, "cluster_labels.nii.gz")
    nib.save(cluster_img, save_img)

    ### GET CLUSTER AVERAGE SIGNAL ###

    logging.info("getting cluster averages")
    all_signals = []
    for z in range(49):
        if isinstance(brain, nib.nifti1.Nifti1Image):
            neural_activity = brain.dataobj[:, :, z, :].reshape(-1, 3384)
        else:
            neural_activity = brain[:, :, z, :].reshape(-1, 3384)
        signals = []
        for cluster_num in range(args.nclusters):
            cluster_indicies = np.where(cluster_labels[z, :] == cluster_num)[0]
            mean_signal = np.mean(neural_activity[cluster_indicies, :], axis=0)
            signals.append(mean_signal)
        signals = np.asarray(signals)
        all_signals.append(signals)
    all_signals = np.asarray(all_signals)
    save_file = os.path.join(cluster_dir, "cluster_signals.npy")
    np.save(save_file, all_signals)
    logging.info("completed clustering")
