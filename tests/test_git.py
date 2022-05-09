import os
from brainsss2.git_utils import get_current_git_hash
import git 


def test_get_current_git_hash():
    return_length=8
    hash = get_current_git_hash(file=__file__)
    assert hash is not None


def test_by_hand():
    script = os.path.realpath(__file__)
    assert script is not None
    assert os.path.exists(script)
    repo = git.Repo(path=script, search_parent_directories=True)
    assert repo.head is not None
    print(repo.head.object.hexsha)
    assert repo.head.object.hexsha is not None
