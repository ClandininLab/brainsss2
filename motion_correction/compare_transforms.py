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

# %% [markdown]
# Comparison of Rigid vs SyN, 
# and different levels of regularization for SyN

# %%
import ants
import numpy as np
import nibabel as nib
import nilearn.plotting
import os
import matplotlib.pyplot as plt
from itertools import combinations
from motcorr_utils import (
    compute_similarity_to_mean,
    compute_timeseries_similarity,
    plot_motpars,
    get_motion_parameters_from_transforms
)
basedir = '/Users/poldrack/data_unsynced/brainsss/ants_example_data'

# %% [markdown]
# ## Setup data
# use the um/mm data to address issues with ants

# %%
origfile = os.path.join(basedir, 'series-tdtomato_size-um_units-mm.nii')
meanfile = os.path.join(basedir, 'series-tdtomato_size-um_units-mm_mean.nii')

origimg = nib.load(origfile)
meanimg = nib.load(meanfile)


print(origimg.affine)
print(origimg.header.get_xyzt_units())

# %%
# create ANTs images
orig_ants = ants.from_nibabel(origimg)
mean_ants = ants.from_nibabel(meanimg)


# %% [markdown]

# ## Test 1: Rigid vs SyN
# We will compare these methods in terms of the relative improvement in 
# similarity to the mean image.  This obviously increases for SyN over Rigid,
# but we'd like to know the relative improvement.


# %%

moco_results = {}

moco_results = {'Rigid': ants.motion_correction(
    image=orig_ants,
    fixed=mean_ants,
    verbose=True,
    type_of_transform='Rigid')}


# %%
# plot motion parameters
motpars = {}
motpars['Rigid'] = get_motion_parameters_from_transforms(
    moco_results['Rigid']['motion_parameters'])[1]
plot_motpars(motpars['Rigid'])

# %%
mean_similarity = {}

mean_similarity['Rigid'] = compute_similarity_to_mean(
    moco_results['Rigid']['motion_corrected'],
    mean_ants)
# also compute similarity of orig data to mean, for comparison
mean_similarity['orig'] = compute_similarity_to_mean(
    orig_ants,
    mean_ants)


# %% [markdown]
# run SyN with default regularization

# %%
moco_results['SyN_default'] = ants.motion_correction(
    image=orig_ants,
    fixed=mean_ants,
    verbose=True,
    type_of_transform='SyN')

# %%
motpars['SyN_default'] = get_motion_parameters_from_transforms(
    moco_results['SyN_default']['motion_parameters'])[1]
plot_motpars(motpars['SyN_default'])

# %%
mean_similarity['SyN_default'] = compute_similarity_to_mean(
    moco_results['SyN_default']['motion_corrected'],
    mean_ants)


# %%
for model in ['orig', 'Rigid', 'SyN_default']:
    plt.plot(mean_similarity[model], 
    label=f'{model}: {np.mean(mean_similarity[model]):.3f}')

plt.title('Similarity to mean image')
plt.legend()

# %% [markdown]

# ## Test 2: SyN with different levels of regularization
# The default is flow_sigma=3, total_sigma=0.
# These are probably too low

# %%
transform = 'SyN' # 'ElasticSyN'
for flow_sigma in [0, 1, 2, 3]:
    for total_sigma in [0, 1, 2, 3]:
        model = f'{transform}_flowsigma-{flow_sigma}_totalsigma-{total_sigma}'
        print('running model:', model)
        moco_results[model] = ants.motion_correction(
            image=orig_ants,
            fixed=mean_ants,
            verbose=True,
            type_of_transform='ElasticSyN',
            flow_sigma=flow_sigma,
            total_sigma=total_sigma)
        mean_similarity[model] = compute_similarity_to_mean(
            moco_results[model]['motion_corrected'],
            mean_ants)
        motpars[model] = get_motion_parameters_from_transforms(
            moco_results[model]['motion_parameters'])[1]


# %%
# compute similarity to mean for each model
similarity_to_mean = {'orig':
    compute_similarity_to_mean(
        orig_ants, mean_ants)}

for img in moco_results:
    if img not in similarity_to_mean:
        similarity_to_mean[img] = compute_similarity_to_mean(
            moco_results[img]['motion_corrected'], mean_ants)

    print(f'{img}: {np.mean(similarity_to_mean[img]):.3f}')

# %%


    
# %%
# display warp fields for each model for a high-motion timepoint

high_motion_timepoint = np.argmax(moco_results['Rigid']['FD'])

warpimgs = {}

for model in moco_results:
    if model == 'Rigid' or 'Elastic' in model:
        continue
    warpimgs[model] = moco_results[model]['motion_parameters'][high_motion_timepoint][0]
    jd = ants.create_jacobian_determinant_image(mean_ants, 
        warpimgs[model]).to_nibabel()
    nilearn.plotting.plot_anat(jd, title=model)

# %%
import scipy.ndimage as ndi

def get_warp_laplacian(warpfile, meanimg):
    """quantify warp roughness using the mean
    absolute laplacian (2nd spatial derivative) of 
    the jacobian determinant of the warp field"""
    warpimg = ants.image_read(warpfile)
    jd = ants.create_jacobian_determinant_image(
        meanimg, warpimg).to_nibabel()
    jd = jd.get_fdata()
    jd = jd / np.max(jd)

    laplacian = ndi.laplace(jd)
    return np.average(np.absolute(laplacian))


roughness = {}
for model in moco_results:
    if model == 'Rigid' or 'Elastic' in model:
        continue
    if model not in roughness:
        warpimgs[model] = moco_results[model]['motion_parameters'][high_motion_timepoint][0]
        roughness[model] = get_warp_laplacian(warpimgs[model], mean_ants)
    print(model, roughness[model])

# %%
# plot roughness in relation to regularization parameters

import pandas as pd

roughness_elastic = {k:v for k,v in roughness.items() 
    if k.find('SyN_flow') == 0}

df = pd.DataFrame({
    'flowsigma': None,
    'totalsigma': None,
    'roughness': None}, 
    index=roughness_elastic.keys())
for k, v in roughness_elastic.items():
    print(k, v)
    df.loc[k, 'flowsigma'] = float(k.split('_')[-2].split('-')[1])
    df.loc[k, 'totalsigma'] = float(k.split('_')[-1].split('-')[1])
    df.loc[k, 'roughness'] = v
# %%
import seaborn as sns
sns.set_palette('viridis')

sns.lineplot(x='flowsigma', y='roughness', hue='totalsigma', 
    data=df.query('flowsigma > 2 & totalsigma > 1'))
# %%
sns.lineplot(hue='flowsigma', y='roughness', x='totalsigma', 
    data=df)

# %%
