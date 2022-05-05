# pyright: reportMissingImports=false
import sys
import argparse
from brainsss2.h5_to_nii import h5_to_nii


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='convert h5 to nii')
    parser.add_argument('-f', '--file', type=str, help='file to process', required=True)
    parser.add_argument('-o', '--outfile', type=str, help='output file')

    args = parser.parse_args(sys.argv[1:])

    assert 'h5' in args.file, 'file must be h5'

    h5_to_nii(args.file, args.outfile)
