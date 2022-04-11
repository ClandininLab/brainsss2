import numpy as np
import sys
import os
import json
import matplotlib.pyplot as plt
from skimage.filters import threshold_triangle
import psutil
#import brainsss
import nibabel as nib
import argparse
from pathlib import Path

def parse_args(input):
    parser = argparse.ArgumentParser(description='run bleaching qc')
    parser.add_argument('-d', '--dir', type=str, 
        help='directory containing func or anat data', required=True)
    parser.add_argument('--verbose', action='store_true', help='verbose output')
    args = parser.parse_args(input)
    return(args)


def load_data(args):
    """determine directory type and load data"""
    files = [f.as_posix() for f in Path(args.dir).glob('*_channel*.nii')]
    data_mean = {}
    for file in files:
        if args.verbose:
            print(f'processing {file}')
        if os.path.exists(file):
            brain = np.asarray(nib.load(file).get_fdata(), dtype='uint16')
            data_mean[file] = np.mean(brain,axis=(0,1,2))
            del brain
        else:
            print(F"Not found (skipping){file:.>{width-20}}")
    return(data_mean)


def get_bleaching_curve(data_mean, args):
    """get bleaching curve"""
    
    plt.rcParams.update({'font.size': 24})
    fig = plt.figure(figsize=(10,10))
    signal_loss = {}
    for file in data_mean:
        xs = np.arange(len(data_mean[file]))
        color='k'
        if file[-1] == '1': color='red'
        if file[-1] == '2': color='green'
        plt.plot(data_mean[file],color=color,label=file)
        linear_fit = np.polyfit(xs, data_mean[file], 1)
        plt.plot(np.poly1d(linear_fit)(xs),color='k',linewidth=3,linestyle='--')
        signal_loss[file] = linear_fit[0]*len(data_mean[file])/linear_fit[1]*-100
    plt.xlabel('Frame Num')
    plt.ylabel('Avg signal')
    loss_string = ''
    for file in data_mean:
        loss_string = loss_string + file + ' lost' + F'{int(signal_loss[file])}' +'%\n'
    plt.title(loss_string, ha='center', va='bottom')

    save_file = os.path.join(args.dir, 'bleaching.png')
    plt.savefig(save_file,dpi=300,bbox_inches='tight')



if __name__ == '__main__':
    args = parse_args(sys.argv[1:])

    #TODO: Fix logging
    # logfile = args['logfile']
    width = 120

    print('loading data')
    data_mean = load_data(args)

    print('getting bleaching curve')
    get_bleaching_curve(data_mean, args)
