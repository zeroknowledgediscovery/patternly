import os
import traceback
import glob
import uuid
import shutil
import atexit
from collections import deque, defaultdict
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

class DirectedGraph:
    def __init__(self, size):
        self.size = size
        self.graph = defaultdict(set)

    def from_matrix(self, matrix, threshold=0):
        if len(matrix) != len(matrix[0]):
            raise ValueError("Matrix must be square.")

        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] >= threshold:
                    self.graph[i].add(j)

    def add(self, node_1, node_2):
        self.graph[node_1].add(node_2)

    def find_scc(self):
        """ Use Tarjan's algorithm to find strongly connected components """

        self.ids = [-1 for _ in range(self.size)]
        self.low_links = [-1 for _ in range(self.size)]
        self.on_stack = [False for _ in range(self.size)]
        self.stack = deque()
        self.curr_id = 0
        self.num_scc = 0
        for node in range(self.size):
            if self.ids[node] == -1:
                self.dfs(node)

        return self.num_scc

    def dfs(self, root_node):
        self.stack.append(root_node)
        self.on_stack[root_node] = True
        self.ids[root_node] = self.curr_id
        self.low_links[root_node] = self.curr_id
        self.curr_id += 1

        for node in self.graph[root_node]:
            if self.ids[node] == -1:
                self.dfs(node)
                self.low_links[root_node] = min(self.low_links[root_node], self.low_links[node])
            elif self.on_stack[node]:
                self.low_links[root_node] = min(self.low_links[root_node], self.ids[node])

        if self.ids[root_node] == self.low_links[root_node]:
            while len(self.stack) > 0:
                self.on_stack[self.stack.pop()] = False
            self.num_scc += 1


