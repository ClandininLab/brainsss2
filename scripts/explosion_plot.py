# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% tags=[]

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import ants
import easydict
from brainsss2.explosion_plot import (
    load_roi_atlas,
    load_explosion_groups,
    unnest_roi_groups,
    make_single_roi_masks,
    make_single_roi_contours,
    place_roi_groups_on_canvas
)
from brainsss2.argparse_utils import get_base_parser

# %% tags=[]
def parse_args(input, allow_unknown=True, require_args=True):
    parser = get_base_parser('explosion visualization')

    # need to add this manually to procesing steps in order to make required

    parser.add_argument('--file', required=require_args,
        type=str, help='nii file to visualize')
    parser.add_argument('--flydir',
        type=str, help='fly dir',
        required=require_args)
    parser.add_argument('--timepoint', type=int, default=0,
        help='timepoint to visualize for 4d files')
    parser.add_argument('--anatfile', type=str,
        default='preproc/anatomy_channel_1_res-2.0mu_moco_mean.nii',
        help='channel 1 mean image for functional data (after moco)')
    parser.add_argument('--outfile', type=str,
        default='explosion_plot.png',
        help='output file (will be expanded for 4d images')
    parser.add_argument('--cmap', type=str, default='hot',
        help='color map to use for plotting')
    atlas = parser.add_argument_group('atlas options')
    atlas.add_argument('--atlasdir',
        type=str,
        help='directory containing atlas files, defaults to <flydir>/../../atlas')
    atlas.add_argument(
        '--atlasfile',
        type=str,
        help='atlas file',
        default='20220301_luke_2_jfrc_affine_zflip_2umiso.nii')
    atlas.add_argument(
        '--atlasroifile',
        type=str,
        help='atlas roi file',
        default='jfrc_2018_rois_improve_reorient_transformed.nii')
    atlas.add_argument(
        '--explosionroifile',
        type=str,
        help='explosion roi file',
        default='20220425_explosion_plot_rois.pickle')

    if allow_unknown:
        args, unknown = parser.parse_known_args()
    else:
        args = parser.parse_args()

    return args


def get_transforms(args):
    warp_directory = os.path.join(args.flydir, "registration/transforms")
    assert os.path.exists(warp_directory)

    transformfiles = {
        'anat_to_atlas_affine': os.path.join(warp_directory, "anat_to_atlas_0GenericAffine.mat"),
        'anat_to_atlas_warp': os.path.join(warp_directory, "anat_to_atlas_1InverseWarp.nii.gz"),
        'func_to_anat_affine': os.path.join(warp_directory, "func_to_anat_0GenericAffine.mat"),
        'func_to_anat_warp': os.path.join(warp_directory, "func_to_anat_1InverseWarp.nii.gz"),
    }

    transforms = {
        'anat_to_atlas': [
            transformfiles['anat_to_atlas_warp'],
            transformfiles['anat_to_atlas_affine']],
        'func_to_anat': [
            transformfiles['func_to_anat_warp'],
            transformfiles['func_to_anat_affine']],
        'func_to_atlas': [
            transformfiles['func_to_anat_affine'],
            transformfiles['func_to_anat_warp'],
            transformfiles['anat_to_atlas_affine'],
            transformfiles['anat_to_atlas_warp']]
    }

    return transforms

# %% tags=[]

def warp_func_to_atlas(args, atlas):
    img_ants = ants.image_read(args.file)

    atlas_ants = ants.from_numpy(atlas)
    atlas_ants.set_spacing(img_ants.spacing)
    atlas_ants.set_direction(img_ants.direction)

    transforms = get_transforms(args)

    func2atlas = ants.apply_transforms(fixed=atlas_ants, moving=img_ants,
                                    transformlist=transforms['func_to_atlas'],
                                    which_to_invert=[True, False, True, False],
                                    interpolator='nearestNeighbor')
    
    return(func2atlas)

# %% tags=[]

def get_atlas(args, flip_z=False):
    atlasroifile = os.path.join(args.atlasdir, args.atlasroifile)
    atlas = load_roi_atlas(atlasroifile)
    if flip_z:
        atlas = np.flip(atlas, axis=-1)
    return(atlas)

# %% tags=[]

def make_explosion_plot(args, func2atlas, atlas):
    explosion_roi_file = os.path.join(args.atlasdir, args.explosionroifile)
    explosion_rois = load_explosion_groups(explosion_roi_file)
    all_rois = unnest_roi_groups(explosion_rois)
    roi_masks = make_single_roi_masks(all_rois, atlas)
    roi_contours = make_single_roi_contours(roi_masks, atlas)

    input_canvas = np.ones((500, 500, 3))  #+.5 #.5 for diverging
    data_to_plot = func2atlas[:, :, ::-1]
    vmax = np.max(data_to_plot)
    explosion_map = place_roi_groups_on_canvas(explosion_rois,
        roi_masks,
        roi_contours,
        data_to_plot,
        input_canvas,
        vmax=vmax,
        cmap=args.cmap)

    plt.figure(figsize=(10, 10))
    plt.imshow(explosion_map)
    plt.ylim(500, 175)
    plt.axis('off')
    plt.title(args.file)
    plt.savefig(args.outfile)

# %% tags=[]
if __name__ == "__main__":

    args = easydict.EasyDict({
        'file': '../notebooks/atlasroi_test.nii',
        'flydir': '/data/brainsss/processed/fly_001',
        'anatfile': 'preproc/anatomy_channel_1_res-2.0mu_moco_mean.nii',
        'outfile': 'explosion_plot.png',
        'cmap': 'hot',
        'atlasdir': '/data/brainsss/atlas',
        'atlasfile': '20220301_luke_2_jfrc_affine_zflip_2umiso.nii',
        'atlasroifile': 'jfrc_2018_rois_improve_reorient_transformed.nii',
        'explosionroifile': '20220425_explosion_plot_rois.pickle'
    })
    # args = parse_args(sys.argv[1:])
    basedir = '/'.join(args.flydir.split('/')[:-2])
    if args.atlasdir is None:
        args.atlasdir = os.path.join(basedir, 'atlas')
        assert os.path.exists(args.atlasdir), 'atlas dir does not exist'

    setattr(args, 'func_path', os.path.join(args.flydir, 'func_0/'))
    assert os.path.exists(args.func_path), f'func dir {args.func_path} does not exist'
    setattr(args, 'anat_path', os.path.join(args.flydir, 'anat_0/'))
    assert os.path.exists(args.anat_path), f'anat dir {args.anat_path} does not exist'

    atlas = get_atlas(args)

    func2atlas = warp_func_to_atlas(args, atlas)

    make_explosion_plot(args, func2atlas, atlas)

# %%
