import os
import shutil
from sysconfig import get_path 


def shortcut_for_quict():
    """ Copy the quict cli into bin folder, after this, allows using quict in Command Line Interface directly. """
    # TODO: Only working for Linux Environment, need test in Windows and Mac
    bin_path = get_path("scripts")
    quict_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../quict.py")

    shutil.copyfile(quict_file_path, f"{bin_path}/quict")
    os.chmod(f"{bin_path}/quict", 0o555)
