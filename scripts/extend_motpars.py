# create extended motion parameters

import sys
import os
from brainsss2.argparse_utils import get_base_parser # noqa
from brainsss2.motion_correction import extend_motpars


def parse_args(input, allow_unknown=True):
    parser = get_base_parser('extend motion parameters')

    parser.add_argument('-f', '--file', type=str,
        help='motion parameter file to process', required=True)
    parser.add_argument('--outfile', type=str,
        help='output file name (defaults to <file>_extended.txt)')
    if allow_unknown:
        args, unknown = parser.parse_known_args()
    else:
        args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    assert os.path.exists(args.file), f"file {args.file} does not exist"

    extend_motpars(args.file, args.outfile)
