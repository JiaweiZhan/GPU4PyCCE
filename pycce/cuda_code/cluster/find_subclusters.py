import numpy as np
import cupy as cp
import os

directory = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(directory, 'find_subclusters.cu')) as f:
    code = f.read()
module = cp.RawModule(code=code)
find_subcluster_func = module.get_function("find_subclusters")

def cupy_unique_axis(sorted_a):
    # Compute a boolean mask where the first entry is True, and an entry is True
    # if it is different from the previous entry.
    mask = cp.ones(len(sorted_a), dtype=bool)
    mask[1:] = (sorted_a[1:] != sorted_a[:-1]).any(axis=1)

    # Index into the sorted array with the mask to get unique rows
    unique_array = sorted_a[mask]

    return unique_array

def find_subclusters_gpu(bonds,
                        batch_size=80000,
                        strong=False):
    flat_bonds = bonds.flatten()
    _, counts = np.unique(flat_bonds, return_counts=True)
    max_degree = np.max(counts)
    bonds = cp.asarray(bonds, dtype=cp.int32)
    bonds = cp.sort(bonds, axis=-1)
    nbond = bonds.shape[0]
    cluster_prev = bonds

    results = []
    block_size = 128
    nbatch = (nbond - 1) // batch_size + 1
    cluster_new = cp.zeros((batch_size * max_degree, 3), dtype=cp.int32)
    index = cp.array([0], cp.int32)
    for i in range(nbatch):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, nbond)
        nbond_b = end_idx - start_idx
        grid_size = (nbond_b - 1) // block_size + 1
        index[0] = 0
        find_subcluster_func((grid_size,), (block_size,),
                             (bonds, cluster_prev[start_idx: end_idx], cluster_new, nbond, nbond_b, cluster_prev.shape[-1], index, strong),
                             shared_mem=block_size * 2 * cp.int32().itemsize)
        results.append(cluster_new[:index[0]])
    results = cp.vstack(results)
    # Stack columns for lexsort in cupy
    keys = cp.stack((results[:, 2], results[:, 1], results[:, 0]))
    sort_idx = cp.lexsort(keys)
    results = results[sort_idx]
    results = cupy_unique_axis(results)
    return results

if __name__ == "__main__":
    from cupyx.profiler import benchmark

    def generate_graph(num_vertices, max_edges_per_vertex):
        '''
        Generate a random graph.

        Parameters:
        - num_vertices: Total number of vertices in the graph.
        - max_edges_per_vertex: Maximum number of edges a single vertex can have.

        Returns:
        - edges: A list of tuples where each tuple represents a bond/edge between two vertices.
        '''

        edges = set()
        for vertex in range(num_vertices):
            num_edges = np.random.randint(1, max_edges_per_vertex + 1)
            for _ in range(num_edges):
                connected_vertex = np.random.randint(0, num_vertices)
                # Ensure an edge doesn't connect a vertex to itself and is not already in the set.
                while connected_vertex == vertex or (min(vertex, connected_vertex), max(vertex, connected_vertex)) in edges:
                    connected_vertex = np.random.randint(0, num_vertices)
                edges.add((min(vertex, connected_vertex), max(vertex, connected_vertex)))
        return list(edges)

    nvertices = 10000
    max_degree = 5
    bonds = generate_graph(nvertices, max_degree)
    bonds = np.array(bonds, dtype=cp.int32)
    nbond = bonds.shape[0]
    print(f"generate {nbond} bonds.")

    print(benchmark(find_subclusters_gpu, (bonds,), n_repeat=20))
