import os
import traceback
import glob
import uuid
import shutil
import atexit

def os_remove(filename: str) -> int:
    try:
        os.remove(filename)
        return 0
    except OSError:
        return 1

def run_once_per_process(f):
    def wrapper(*args, **kwargs):
        if os.getpgrp() != wrapper.has_run:
            wrapper.has_run = os.getpgrp()
            return f(*args, **kwargs)

    wrapper.has_run = 0
    return wrapper

def _clean_up_temp_folder(path, signal=None, frame=None):
    if signal is not None:
        traceback.print_stack(frame)
    try:
        [
            os_remove(x) if os.path.isfile(x) else shutil.rmtree(x)
            for x in glob.glob(path + "*")
        ]
        os.rmdir(os.path.dirname(path))
    except OSError:
        pass

@run_once_per_process
def add_signal(path):
    atexit.register(_clean_up_temp_folder, path)

def RANDOM_NAME(clean: bool = True, path: str = "zed_temp") -> str:
    if not os.path.isdir(path):
        os.mkdir(path)
    path = os.path.join(path, "clean_")
    if clean:
        # path = path + str(os.getpgrp()) + "_"
        add_signal(path)
    random_name = str(uuid.uuid4())
    full = path + random_name
    while os.path.isfile(full):
        # print("name_double")
        random_name = str(uuid.uuid4())
        full = path + random_name

    return full