import os
import traceback
import glob
import uuid
import shutil
import atexit
import numpy as np

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

class UnionFind:
    def __init__(self, size):
        if size <= 0:
            raise ValueError("Size must be at least 1.")
        self.size = size
        self.n_components = size
        self.roots = np.arange(size, dtype=np.int32)
        self.component_sizes = np.full(shape=size, fill_value=1, dtype=np.int32)

    def find(self, node):
        root = node
        while (root != self.roots[root]):
            root = self.roots[root]
        return root

    def union(self, node_1, node_2, ranks=None):
        root_1 = self.find(node_1)
        root_2 = self.find(node_2)
        if root_1 == root_2:
            return

        if ranks is not None and len(ranks) == self.size:
            main_root = root_1 if ranks[root_1] > ranks[root_2] else root_2
            sub_root = root_2 if ranks[root_1] > ranks[root_2] else root_1
        else:
            main_root = root_1 if self.component_sizes[root_1] > self.component_sizes[root_2] else root_2
            sub_root = root_2 if self.component_sizes[root_1] > self.component_sizes[root_2] else root_1

        self.roots[sub_root] = main_root
        self.compress(node_1, main_root)
        self.compress(node_2, main_root)
        self.component_sizes[main_root] += self.component_sizes[sub_root]
        self.n_components -= 1

        return self

    def compress(self, node, root=None):
        if root is None:
            root = self.find(node)

        while (node != root):
            next_node = self.roots[node]
            self.roots[node] = root
            node = next_node

    def compress_all(self):
        for i in range(self.size):
            self.compress(i)
        return self

    def connected(self, node_1, node_2):
        return self.find(node_1) == self.find(node_2)

