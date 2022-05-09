# git utils

import os
import git


def get_current_git_hash(file=None, return_length=8):
    try:
        if file is None:
            file = __file__
        script = os.path.realpath(file)
        print(script)
        repo = git.Repo(path=script, search_parent_directories=True)
        print(repo)
        return repo.head.object.hexsha[:return_length]

    except Exception as e:
        print(f'unable to obtain git hash: {e}')
        return None
