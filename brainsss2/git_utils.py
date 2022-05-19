# git utils

import os
import git


def get_current_git_hash(file=None, return_length=8):
    try:
        if file is None:
            file = __file__
        script = os.path.realpath(file)
        repo = git.Repo(path=script, search_parent_directories=True)
        return repo.head.object.hexsha[:return_length]
    except:  # noqa
        return None
