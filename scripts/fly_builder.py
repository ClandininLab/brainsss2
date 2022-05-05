# main script to import and build processed fly dirs

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import os
import sys
from brainsss2.logging_utils import setup_logging, remove_existing_file_handlers # noqa
from brainsss2.utils import get_resolution, load_timestamps, sort_nicely # noqa
from brainsss2.fly_builder import parse_args, build_fly


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    assert os.path.exists(args.basedir), f"basedir {args.basedir} does not exist"

    print(args)
    _ = remove_existing_file_handlers()
    args = setup_logging(args, 'flybuilder')

    build_fly(args)
