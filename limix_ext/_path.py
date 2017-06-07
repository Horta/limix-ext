from __future__ import print_function
import contextlib
import errno
import tempfile
import shutil
import os
import subprocess
import hashlib
import sys
from distutils.spawn import find_executable


@contextlib.contextmanager
def temp_folder():
    folder = tempfile.mkdtemp()
    try:
        yield folder
    finally:
        shutil.rmtree(folder)

def count_lines(filepath):
    return sum(1 for line in open(filepath, 'r'))

class TmpFileCopy(object):
    """Creates a temporary file copy."""
    def __init__(self, path, copy_back=False):
        self._dst = None
        self._path = path
        self._folder = None
        self._copy_back = copy_back
        if not os.path.exists(path):
            raise Exception('%s does not exist.' % path)

    def __enter__(self):
        self._folder = tempfile.mkdtemp()
        bname = os.path.basename(self._path)
        dst = os.path.join(self._folder, bname)

        retcode = subprocess.call("cp " + self._path + " " + dst,
                                  shell=True)

        if retcode < 0:
            print("Child was terminated by signal %d" % retcode,
                  file=sys.stderr)
            raise Exception('Could not copy %s.' % self._path)

        self._dst = dst
        return dst

    def __exit__(self, *args):
        if self._copy_back:
            retcode = subprocess.call("cp " + self._dst + " " + self._path,
                                      shell=True)

            if retcode < 0:
                print("Child was terminated by signal %d" % retcode,
                      file=sys.stderr)
                raise Exception('Could not copy %s back.' % self._dst)
        shutil.rmtree(self._folder)

class ChDir(object):
    """Steps into a directory temporarily.
    """
    def __init__(self, path):
        self.old_dir = os.getcwd()
        self.new_dir = path

    def __enter__(self):
        os.chdir(self.new_dir)

    def __exit__(self, *args):
        os.chdir(self.old_dir)

def make_sure_path_exists(path):
    """Creates a path recursively if necessary.
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise

def cp(folder_src, folder_dst):
    """Uses cp shell command to copy files from a folder to another.
    """
    retcode = subprocess.call("cp " + folder_src + "/* " + folder_dst,
                              shell=True)

    if retcode < 0:
        print("Child was terminated by signal %d" % retcode,
              file=sys.stderr)

def rrm(paths):
    """Uses rm shell command to remove files.
    """
    paths = paths if isinstance(paths, list) else [paths]
    if len(paths) == 0:
        return

    cmd = subprocess.list2cmdline(['rm', '-rf'] + paths)
    retcode = subprocess.call(cmd, shell=True)

    if retcode < 0:
        print("Child was terminated by signal %d" % retcode,
              file=sys.stderr)

def touch(fname, times=None):
    """Creates an empty file.
    """
    with open(fname, 'a'):
        os.utime(fname, times)

def bin_exists(name):
    """Checks whether an executable file exists.
    """
    return find_executable(name) is not None

def folder_hash(folder, exclude_files=None):
    """Recursively compute the hash of all files in a folder and sum it up.
    """
    if exclude_files is None:
        exclude_files = []

    if not bin_exists('md5deep'):
        raise EnvironmentError("Couldn't not find md5deep.")

    out = subprocess.check_output('md5deep -r %s' % folder, shell=True)
    lines = sorted(out.strip(b'\n').split(b'\n'))

    m = hashlib.md5()
    for line in lines:
        hash_ = line[0:32]
        fp = line[34:]
        if os.path.basename(fp) not in exclude_files:
            m.update(hash_)
    return m.hexdigest()
