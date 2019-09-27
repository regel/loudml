# Definition of the version number
import os
from io import open as io_open

__all__ = ["__version__"]

# major, minor, patch, -extra
version_info = 1, 6, 0

# Nice string for the version
__version__ = '.'.join(map(str, version_info))


# auto -extra based on commit hash (if not tagged as release)
try:
    scriptdir = os.path.dirname(__file__)
except NameError:  # We are in setup.py script, not a module
    import sys
    scriptdir = os.path.dirname(os.path.abspath(sys.argv[0]))

gitdir = os.path.abspath(os.path.join(scriptdir, "..", ".git"))
if os.path.isdir(gitdir):  # pragma: nocover
    extra = None
    # Open config file to check if we are in loudml project
    with io_open(os.path.join(gitdir, "config"), 'r') as fh_config:
        if 'loudml' in fh_config.read():
            # Open the HEAD file
            with io_open(os.path.join(gitdir, "HEAD"), 'r') as fh_head:
                extra = fh_head.readline().strip()
            # in a branch => HEAD points to file containing last commit
            if 'ref:' in extra:
                # reference file path
                ref_file = extra[5:]
                branch_name = ref_file.rsplit('/', 1)[-1]

                ref_file_path = os.path.abspath(os.path.join(gitdir, ref_file))
                # check that we are in git folder
                # (by stripping the git folder from the ref file path)
                if os.path.relpath(
                        ref_file_path, gitdir).replace('\\', '/') != ref_file:
                    # out of git folder
                    extra = None
                else:
                    # open the ref file
                    with io_open(ref_file_path, 'r') as fh_branch:
                        commit_hash = fh_branch.readline().strip()
                        extra = commit_hash[:8]
                        if branch_name != "master":
                            extra += '.' + branch_name

            # detached HEAD mode, already have commit hash
            else:
                extra = extra[:8]

    # Append commit hash (and branch) to version string if not tagged
    if extra is not None:
        try:
            with io_open(os.path.join(gitdir, "refs", "tags",
                                      'v' + __version__)) as fdv:
                if fdv.readline().strip()[:8] != extra[:8]:
                    __version__ += '-' + extra
        except FileNotFoundError:
            __version__ += '-' + extra[:8]
