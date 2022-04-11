# git utils

import os
import git


def get_current_git_hash(return_length=8):
    script = os.path.realpath(__file__)
    repo = git.Repo(path=script, search_parent_directories=True)
    return(repo.head.object.hexsha[:return_length])
