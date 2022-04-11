
import ants
import os 
import numpy as np
from brainsss.utils import get_resolution


def ants_affine_to_distance(affine):
    """ by @dangom, https://github.com/ANTsX/ANTsPy/issues/71"""
    dx, dy, dz = affine[9:]

    rot_x = np.arcsin(affine[6])
    cos_rot_x = np.cos(rot_x)
    rot_y = np.arctan2(affine[7] / cos_rot_x, affine[8] / cos_rot_x)
    rot_z = np.arctan2(affine[3] / cos_rot_x, affine[0] / cos_rot_x)

    deg = np.degrees

    return dx, dy, dz, deg(rot_x), deg(rot_y), deg(rot_z)


def get_dataset_resolution(datadir):
    """get the resolution of the dataset"""
    xmlfiles = [i for i in os.listdir(datadir) if i.endswith('.xml')]
    xmlfile = os.path.join(datadir, 'functional.xml') if 'functional.xml' in xmlfiles else 'functional.xml'
    assert os.path.exists(xmlfile), 'xml file not found'

    resolution = get_resolution(xmlfile)
    return resolution


def get_motion_parameters_from_transforms(transformlist, resolution):
    """only gets rigid body transform parameters
    
    transformlist: list of transform files
    resolution: resolution of the image (in microns)"""
    transform_parameters = np.zeros((len(transformlist), 12))
    motion_parameters = np.zeros((len(transformlist), 6))
    for idx, t in enumerate(transformlist):
        if isinstance(t, list):
            paramfile = [i for i in t if i.endswith('0GenericAffine.mat')][0]
        elif t.endswith('0GenericAffine.mat'):
            paramfile = t
        else:
            raise ValueError('transformlist must contain ANTS affine matrix files')
        temp = ants.read_transform(paramfile)
        transform_parameters[idx, :] = temp.parameters
        motion_parameters[idx, :] = ants_affine_to_distance(temp.parameters)
    return transform_parameters, motion_parameters


