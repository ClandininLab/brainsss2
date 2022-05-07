# git utils

import os
import git


def get_current_git_hash(return_length=8):
    try:
        script = os.path.realpath(__file__)
        repo = git.Repo(path=script, search_parent_directories=True)
        return repo.head.object.hexsha[:return_length]

    except Exception as e:
        print(f'unable to obtain git hash: {e}')
        return None
