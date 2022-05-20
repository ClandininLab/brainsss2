import numpy as np


def get_transformed_data_slice(zdata, zmask):
    if zmask.sum() == 0:
        return(None, zmask)
    zmask_vec = zmask.reshape(np.prod(zmask.shape))
    zdata_mat = zdata.reshape((np.prod(zmask.shape), zdata.shape[-1]))
    return(zdata_mat[zmask_vec == 1], zmask_vec)
