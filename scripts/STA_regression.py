import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
import h5py
import json
import nibabel as nib
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import r2_score
from brainsss2.visual import (
    load_photodiode,
    get_stimulus_metadata,
    extract_stim_times_from_pd,
)
from brainsss2.fictrac import smooth_and_interp_fictrac, load_fictrac
from brainsss2.brain_utils import (
    extract_traces,
    get_visually_evoked_turns,
    load_fda_meanbrain,
    make_STA_brain,
    STA_supervoxel_to_full_res,
    warp_STA_brain,
)
from brainsss2.explosion_plot import (
    load_roi_atlas,
    load_explosion_groups,
    unnest_roi_groups,
    make_single_roi_masks,
    make_single_roi_contours,
    place_roi_groups_on_canvas,
)
from brainsss2.argparse_utils import get_base_parser
from brainsss2.utils import load_timestamps


def parse_args(input, allow_unknown=True):
    parser = get_base_parser("stimulus-triggered averaging")

    parser.add_argument(
        "-d", "--dir", type=str, help="func directory to be analyzed", required=True
    )
    parser.add_argument("--atlasdir", type=str, help="atlas directory", required=True)
    parser.add_argument(
        "--atlasfile", type=str, default="20220301_luke_2_jfrc_affine_zflip_2umiso.nii"
    )
    parser.add_argument(
        "--roifile", type=str, default="jfrc_2018_rois_improve_reorient_transformed.nii"
    )
    parser.add_argument(
        "--explosionroifile", type=str, default="20220425_explosion_plot_rois.pickle"
    )
    parser.add_argument('--ntimepoints', type=int, default=20,
        help='number of post-onset timepoints to estimate response')
    parser.add_argument('--nprestim', type=int, default=4,
        help='number of pre-onset timepoints to estimate response')
    if allow_unknown:
        args, unknown = parser.parse_known_args()
    else:
        args = parser.parse_args()
    return args


def prep_visual_stimuli(func_path):

    vision_path = os.path.join(func_path, "visual")

    t, ft_triggers, pd1, pd2 = load_photodiode(vision_path)
    stimulus_start_times = extract_stim_times_from_pd(pd2, t)

    stim_ids, angles = get_stimulus_metadata(vision_path)
    print(f"Found {len(stim_ids)} presented stimuli.")

    # *100 puts in units of 10ms, which will match fictrac
    starts_angle = {}
    for angle in [0, 180]:
        starts_angle[angle] = [
            int(stimulus_start_times[i] * 100)
            for i in range(len(stimulus_start_times))
            if angles[i] == angle
        ]
    print(
        f"starts_angle_0: {len(starts_angle[0])}. starts_angle_180: {len(starts_angle[180])}"
    )

    # get 1ms version to match neural timestamps
    starts_angle_ms = {}
    for angle in [0, 180]:
        starts_angle_ms[angle] = [i * 10 for i in starts_angle[angle]]
    return (starts_angle, starts_angle_ms)


def prep_fictrac(func_path):
    fictrac_path = os.path.join(func_path, "fictrac")
    fictrac_raw = load_fictrac(fictrac_path)

    fps = 100
    resolution = 10  # desired resolution in ms
    expt_len = fictrac_raw.shape[0] / fps * 1000
    behaviors = ["dRotLabY", "dRotLabZ"]
    fictrac = {}
    for behavior in behaviors:
        if behavior == "dRotLabY":
            short = "Y"
        elif behavior == "dRotLabZ":
            short = "Z"
        fictrac[short] = smooth_and_interp_fictrac(
            fictrac_raw, fps, resolution, expt_len, behavior
        )
    # fictrac_timestamps = np.arange(0,expt_len,resolution)
    return fictrac


def extract_STB(fictrac, starts_angle):
    pre_window = 200
    post_window = 300

    behavior_traces = {}
    mean_trace = {}
    sem_trace = {}
    for angle in [0, 180]:
        behavior_traces[angle], mean_trace[angle], sem_trace[angle] = extract_traces(
            fictrac, starts_angle[angle], pre_window, post_window
        )
    return (behavior_traces, mean_trace, sem_trace)


def plot_stb():
    plt.figure(figsize=(10, 10))

    for angle, color in zip([0, 180], ["blue", "red"]):
        plt.plot(mean_trace[angle], color=color, linewidth=3)
        plt.fill_between(
            np.arange(len(mean_trace[angle])),
            mean_trace[angle] - sem_trace[angle],
            mean_trace[angle] + sem_trace[angle],
            color=color,
            alpha=0.3,
        )
    for line in [200, 250, 300]:
        plt.axvline(line, color="k", linestyle="--", lw=2)


def extract_visually_evoked_turns(behavior_traces, mean_trace, starts_angle_ms):
    ve_turns = {}
    ve_turn_times = {}
    for angle, direction in zip([0, 180], ["neg", "pos"]):
        ve_turns[angle], ve_turn_times[angle] = get_visually_evoked_turns(
            behavior_traces[angle],
            mean_trace[angle],
            start=250,
            stop=300,
            r_thresh=0.3,
            av_thresh=50,
            stim_times=starts_angle_ms[angle],
            expected_direction=direction,
        )
    ve_no_turns = {}
    ve_no_turn_times = {}
    for angle in [0, 180]:
        ve_no_turns[angle], ve_no_turn_times[angle] = get_stimuli_where_no_behavior(
            behavior_traces[angle],
            start=250,
            stop=300,
            num_traces_to_return=len(
                ve_turns[angle]
            ),  # get the same number as ve_turns
            stim_times=starts_angle_ms[angle],
        )
    return (ve_turns, ve_turn_times, ve_no_turns, ve_no_turn_times)


def plot_ve_turns():
    # fig = plt.figure(figsize=(10,10))

    plt.subplot(211)
    plt.imshow(ve_turns[180], aspect=5, cmap="seismic", vmin=-300, vmax=300)
    for line in [200, 250, 300]:
        plt.axvline(line, color="k", linestyle="--", lw=2)

    plt.subplot(212)
    plt.imshow(ve_turns[0], aspect=5, cmap="seismic", vmin=-300, vmax=300)
    for line in [200, 250, 300]:
        plt.axvline(line, color="k", linestyle="--", lw=2)

    # +
    plt.figure(figsize=(10, 10))

    for angle, color in zip([0, 180], ["blue", "red"]):
        mean_ve_trace = np.mean(ve_turns[angle], axis=0)
        sem_ve_trace = scipy.stats.sem(ve_turns[angle], axis=0)

        plt.plot(mean_ve_trace, color=color, linewidth=3)
        plt.fill_between(
            np.arange(len(mean_ve_trace)),
            mean_ve_trace - sem_ve_trace,
            mean_ve_trace + sem_ve_trace,
            color=color,
            alpha=0.3,
        )
    for line in [200, 250, 300]:
        plt.axvline(line, color="k", linestyle="--", lw=2)


def get_stimuli_where_no_behavior(
    traces, start, stop, num_traces_to_return, stim_times
):
    amount_of_behavior = np.mean(np.abs(traces[:, start:stop]), axis=-1)
    indicies = np.argsort(amount_of_behavior)
    top_x_indicies = indicies[:num_traces_to_return]
    return traces[top_x_indicies, :], np.asarray(stim_times)[top_x_indicies]


def old_STA():

    all_explosions = {}
    for condition in ["ve_no_0", "ve_no_180", "ve_0", "ve_180"]:
        print(condition)

        if "180" in condition:
            angle = 180
        else:
            angle = 0
        if "no" in condition:
            event_times_list = ve_no_turn_times[angle]
        else:
            event_times_list = ve_turn_times[angle]

        t0 = time.time()
        STA_brain = make_STA_brain(
            neural_signals=all_signals,
            neural_timestamps=timestamps,
            event_times_list=event_times_list,
            neural_bins=neural_bins,
        )
        print(f"STA {time.time()-t0}")

        reformed_STA_brain = STA_supervoxel_to_full_res(STA_brain, cluster_labels)
        #  Axis 1 is timepoints in the average (I think!)
        tempfilt_STA_brain = gaussian_filter1d(
            reformed_STA_brain, sigma=1, axis=1, truncate=1
        )

        t0 = time.time()
        warps = warp_STA_brain(
            STA_brain=tempfilt_STA_brain,
            fly=fly,
            fixed=fixed,
            anat_to_mean_type="myr",  # WHAT IS THIS ABOUT?
            dataset_path=basedir,
        )
        print(f"Warps {time.time()-t0}")

        explosions = []
        t0 = time.time()
        for tp in range(24):
            input_canvas = np.ones((500, 500, 3))  # +.5 #.5 for diverging
            data_to_plot = warps[tp][:, :, ::-1]
            vmax = 0.5
            explosion_map = place_roi_groups_on_canvas(
                explosion_rois,
                roi_masks,
                roi_contours,
                data_to_plot,
                input_canvas,
                vmax=vmax,
                cmap="hot",
            )
            explosions.append(explosion_map)
        print(f"Explosion {time.time()-t0}")
        all_explosions[condition] = explosions
    return all_explosions

    plt.figure(figsize=(10, 10))
    plt.imshow(all_explosions["ve_0"][4][170:, :])  # this was made with cmap=hot

    all_explosions.keys()


# following are for the new STA approach
def get_resampled_slice_data(
    all_signals, event_times_list, slice, scale_data=True, interpolation_type="linear"
):

    slicedata = all_signals[slice, :, :].squeeze().T
    if scale_data:
        slicedata = slicedata - np.mean(slicedata, axis=0)
        slicedata = slicedata / np.std(slicedata, axis=0)
        slicedata = np.nan_to_num(slicedata)

    all_signals_slice_df = pd.DataFrame(
        slicedata, index=pd.to_datetime(timestamps[:, 0] / 1000, unit="s")
    )

    resampled_df = (
        all_signals_slice_df.resample("100ms")
        .mean()
        .interpolate(method=interpolation_type)
    )

    event_times = pd.to_datetime(np.array(event_times_list) / 1000, unit="s")
    event_df = pd.DataFrame(
        {"event": np.zeros(resampled_df.shape[0])}, index=resampled_df.index
    )
    event_df = pd.concat(
        (
            event_df,
            pd.DataFrame({"event": np.ones(event_times.shape[0])}, index=event_times),
        )
    )
    event_df_resampled = event_df.resample(
        "100ms", origin=all_signals_slice_df.index[0]
    ).sum()

    assert event_df_resampled.shape[0] == resampled_df.shape[0]
    return (resampled_df, event_df_resampled)


# +
def make_fir_desmtx(event_df, npts=20, n_prestim_pts=4, fit_intercept=False):
    """
    make a finite impulse response design matrix

    event_vec: data farme
        data frame with one column named "event" and with a datetime index
    npts: int
        number of timepoints after onset
    n_prestim_pts: int
        number of timepoints before onset
    fit_intercept: bool
        whether to include intercept in model

    returns:

    desmtx: numpy ndarray
        design matrix
    """
    event_vec = event_df.event.values
    timedelta = (event_df.index[1] - event_df.index[0]).microseconds / 1000
    timepoints = []
    desmtx = np.zeros((event_vec.shape[0], npts + n_prestim_pts))
    ctr = 0
    for i in range(-n_prestim_pts, 0, 1):
        desmtx[:i, ctr] = event_vec[(-i):]
        timepoints.append(i * timedelta)
        ctr += 1
    desmtx[:, ctr] = event_vec
    timepoints.append(0)
    ctr += 1
    for i in range(1, npts):
        desmtx[i:, ctr] = event_vec[:-i]
        timepoints.append(i * timedelta)
        ctr += 1
    if fit_intercept:
        desmtx = np.hstack((desmtx, np.ones(desmtx.shape[0])[:, np.newaxis]))

    desmtx_df = pd.DataFrame(desmtx, columns=timepoints)
    return desmtx_df


def run_fir_regression(resampled_df, desmtx_df, zero_onset=False):
    lr = LinearRegression()
    X = desmtx_df.values
    y = resampled_df.values
    lr.fit(X, y)
    coefs = lr.coef_

    if zero_onset:
        coefs = coefs - coefs[np.where(desmtx_df.columns == 0.0)[0][0]]

    predictions = lr.predict(X)
    residuals = y - predictions
    squared_resids = (residuals)**2
    df = X.shape[0] - X.shape[1]
    MSE = squared_resids.sum(axis=0) / (df)
    XtX = np.dot(X.T, X)
    tstat = np.zeros(coefs.shape)
    for i in range(coefs.shape[1]):
        # use unscaled coefs for computing t
        tstat[:, i] = lr.coef_[:, i] / np.sqrt(MSE[i] / np.diag(XtX)[i])
        # compute two-tailed p-value
    pval = (1 - scipy.stats.t.cdf(x=np.abs(tstat), df=df)) * 2
    # set pvals for coefs that are nan to 1
    pval = np.nan_to_num(pval, nan=1)
    assert np.min(pval) >= 0
    assert np.max(pval) <= 1

    rsquared = r2_score(y, predictions, multioutput="raw_values")
    return (coefs, rsquared, tstat, pval)


def supervoxels_to_img(supervoxel_data, cluster_img):
    """
    takes in a supervoxel matrix (either (nslices X nclusters or
    nslices X ncluster X ntimepoints) and projects it into
    a nibabel img

    Args:

    supervoxel_data: numpy ndarray
        supervoxel matrix
    cluster_img: nibabel img
        img with the same shape as supervoxel_data with cluster labelings

    returns:
    img: nibabel img
    """
    if len(supervoxel_data.shape) == 2:
        supervoxel_data = np.expand_dims(supervoxel_data, axis=2)
    assert supervoxel_data.shape[0] == cluster_img.shape[-1]
    output_shape = cluster_img.shape + (supervoxel_data.shape[-1],)
    output_img = nib.Nifti1Image(
        np.zeros(output_shape), cluster_img.affine, cluster_img.header
    )
    clusterdata = cluster_img.get_fdata().astype(int)

    for slice in range(supervoxel_data.shape[0]):
        slicedata = np.zeros(cluster_img.shape[:2])
        slicemask = clusterdata[:, :, slice]
        for timepoint in range(supervoxel_data.shape[-1]):
            for cluster in np.unique(slicemask):
                slicedata[slicemask == cluster] = supervoxel_data[
                    slice, cluster, timepoint
                ]
            output_img.dataobj[:, :, slice, timepoint] = slicedata
    return output_img


if __name__ == "__main__":

    args = parse_args(sys.argv[1:])
    STA_path = os.path.join(args.dir, "STA")
    if not os.path.exists(STA_path):
        os.mkdir(STA_path)

    starts_angle, starts_angle_ms = prep_visual_stimuli(args.dir)

    fictrac = prep_fictrac(args.dir)

    behavior_traces, mean_trace, sem_trace = extract_STB(fictrac, starts_angle)

    ve_turns, ve_turn_times, ve_no_turns, ve_no_turn_times = extract_visually_evoked_turns(
        behavior_traces, mean_trace, starts_angle_ms
    )

    cluster_dir = os.path.join(args.dir, "clustering")

    cluster_label_imgfile = os.path.join(cluster_dir, "cluster_labels.nii.gz")
    cluster_label_img = nib.load(cluster_label_imgfile)

    load_file = os.path.join(cluster_dir, "cluster_signals.npy")
    all_signals = np.load(load_file)

    timestamps = load_timestamps(os.path.join(args.dir, "imaging"))

    atlasfile = os.path.join(args.atlasdir, args.atlasfile)
    fixed = load_fda_meanbrain(atlasfile)

    atlasroifile = os.path.join(args.atlasdir, args.roifile)
    atlas = load_roi_atlas(atlasroifile)

    explosion_roi_file = os.path.join(args.atlasdir, args.explosionroifile)
    explosion_rois = load_explosion_groups(explosion_roi_file)
    all_rois = unnest_roi_groups(explosion_rois)
    roi_masks = make_single_roi_masks(all_rois, atlas)
    roi_contours = make_single_roi_contours(roi_masks, atlas)

    # bin_start = -500; bin_end = 2000; bin_size = 100
    # neural_bins = np.arange(bin_start,bin_end,bin_size)

    output_filename = os.path.join(STA_path, "STA_results.h5")
    coef_group = outfile.create_group("coefs")
    r2_group = outfile.create_group("rsquared")

    for condition in ["ve_no_0", "ve_no_180", "ve_0", "ve_180"]:
        if "180" in condition:
            angle = 180
        else:
            angle = 0
        if "no" in condition:
            event_times_list = ve_no_turn_times[angle]
        else:
            event_times_list = ve_turn_times[angle]

        all_coefs = None
        all_pvals = None
        all_rsquared = None
        for slice in range(all_signals.shape[0]):
            resampled_df, event_df_resampled = get_resampled_slice_data(
                all_signals, event_times_list, slice, interpolation_type="linear"
            )

            desmtx_df = make_fir_desmtx(event_df_resampled,
                args.ntimepoints, args.nprestim)

            coefs, rsquared, tstat, pval = run_fir_regression(
                resampled_df, desmtx_df)

            if all_coefs is None:
                all_coefs = np.zeros(
                    (all_signals.shape[0], coefs.shape[0], coefs.shape[1])
                )
                all_pvals = np.zeros(
                    (all_signals.shape[0], coefs.shape[0], coefs.shape[1])
                )
            all_coefs[slice, :, :] = coefs
            all_pvals[slice, :, :] = pval
            print(f"Slice {slice}: {np.max(coefs)}")
            if all_rsquared is None:
                all_rsquared = np.zeros((all_coefs.shape[0], all_coefs.shape[1]))
            all_rsquared[slice, :] = rsquared
        coef_group.create_dataset(condition, data=all_coefs)
        r2_group.create_dataset(condition, data=all_rsquared)
        img = supervoxels_to_img(all_coefs, cluster_label_img)
        img.to_filename(os.path.join(STA_path, f"STA_coefs_{condition}.nii.gz"))
        img = supervoxels_to_img(1 - all_pvals, cluster_label_img)
        img.to_filename(os.path.join(STA_path, f"STA_1-pval_{condition}.nii.gz"))
        img = supervoxels_to_img(all_rsquared, cluster_label_img)
        img.to_filename(os.path.join(STA_path, f"STA_rsquared_{condition}.nii.gz"))
    setattr(args, 'timepoints', list(desmtx_df.columns))
    with open(os.path.join(STA_path, 'STA_settings.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
