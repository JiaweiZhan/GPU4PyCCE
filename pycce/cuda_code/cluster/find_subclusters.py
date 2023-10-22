import numpy as np
import cupy as cp
import os
import scipy
from scipy.sparse import csr_matrix
from pycce.find_clusters import find_subclusters_cpu
from pycce.logging import get_logger

logger = get_logger(__file__)

def cupy_unique_axis(sorted_a: cp.ndarray,
                     ):
    # Compute a boolean mask where the first entry is True, and an entry is True
    # if it is different from the previous entry.
    mask = cp.ones(len(sorted_a), dtype=bool)
    mask[1:] = (sorted_a[1:] != sorted_a[:-1]).any(axis=1)

    # Index into the sorted array with the mask to get unique rows
    unique_array = sorted_a[mask]

    return unique_array

def find_subclusters_k_gpu(bonds: cp.ndarray,
                           prev_clusters: cp.ndarray,
                           batch_size: int=80000,
                           strong: bool=False,
                           ):

    directory = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(directory, 'find_subclusters.cu')) as f:
        code = f.read()
    module = cp.RawModule(code=code)
    find_subcluster_func = module.get_function("find_subclusters")

    flat_bonds = bonds.flatten()
    _, counts = cp.unique(flat_bonds, return_counts=True)
    max_degree = cp.max(counts).item()
    nbond = bonds.shape[0]

    ncluster = prev_clusters.shape[0]
    scluster = prev_clusters.shape[1]

    results = []
    block_size = 128
    nbatch = (ncluster - 1) // batch_size + 1
    cluster_new = cp.zeros((batch_size * max_degree, 3), dtype=cp.int32)
    index = cp.array([0], cp.int32)
    for i in range(nbatch):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, ncluster)
        ncluster_b = end_idx - start_idx
        grid_size = (ncluster_b - 1) // block_size + 1
        index[0] = 0
        find_subcluster_func((grid_size,),
                             (block_size,),
                             (bonds,
                              prev_clusters[start_idx: end_idx],
                              cluster_new,
                              nbond,
                              ncluster_b,
                              scluster,
                              index,
                              strong),
                             shared_mem=block_size * 2 * cp.int32().itemsize)
        results.append(cluster_new[:index[0]])
    results = cp.vstack(results)
    # Stack columns for lexsort in cupy
    keys = []
    for i in reversed(range(scluster + 1)):
        keys.append(results[:, i])
    keys = cp.stack(keys)
    sort_idx = cp.lexsort(keys)
    results = results[sort_idx]
    results = cupy_unique_axis(results)
    return results

def find_subclusters_gpu(maximum_order: int,
                         graph: csr_matrix,
                         labels: np.ndarray,
                         n_components: int,
                         strong: bool=False,
                         ):
    """
    Find subclusters from connectivity matrix.

    Args:
        maximum_order (int):
            Maximum size of the clusters to find.
        graph (csr_matrix): Connectivity matrix.
        labels (ndarray with shape (n,)): Array of labels of the connected components.
        n_components (int): The number of connected components n.
        strong (bool): Whether to find only completely interconnected clusters (default False).

    Returns:
        dict:
            Dictionary with keys corresponding to size of the cluster,
            and value corresponds to ndarray of shape (matrix, N).
            Here matrix is the number of clusters of given size, N is the size of the cluster.
            Each row contains indexes of the bath spins included in the given cluster.
    """
    # bool 1D array which is true when given element of graph corresponds to
    # cluster component
    clusters = {}
    for k in range(1, maximum_order + 1):
        clusters[k] = []
    # print('Number of disjointed clusters is {}'.format(n_components))
    for component in range(n_components):
        vert_pos = (labels == component)
        vertices = cp.asarray(np.nonzero(vert_pos)[0])

        # print('{} cluster contains {} components'.format(component, ncomp))

        # if ncomp <= CCE_order:
        #
        #     clusters[ncomp].append(vertices[np.newaxis, :])
        #
        # else:

        subclusters = {1: vertices[:, np.newaxis]}

        clusters[1].append(vertices[:, np.newaxis])

        if vertices.size >= 2 and maximum_order > 1:
            # Retrieve upper right triangle (remove i,j pairs with i>j),
            # choose only rows corresponding to vertices in the subcluster
            csrmat = scipy.sparse.triu(graph, k=0, format='csr')[vertices.get()]
            # Change to coordinate format of matrix
            coomat = csrmat.tocoo()
            # rows, col give row and colum indexes, which correspond to
            # edges of the graph. as we already slised out the rows,
            # to obtain correct row indexes we need to use vertices array
            row_ind, col_ind = vertices[coomat.row], cp.asarray(coomat.col)

            bonds = cp.asarray(cp.column_stack([row_ind, col_ind]), dtype=cp.int32)
            subclusters[2] = bonds
            clusters[2].append(bonds)

            # Check if [1,2] row in a matrix(Nx2):  any(np.equal(a, [1, 2]).all(1))

            for order in range(3, maximum_order + 1):

                prevClusters = subclusters[order - 1]
                try:
                    subclusters[order] = find_subclusters_k_gpu(bonds, prevClusters, strong=strong)
                    clusters[order].append(subclusters[order])
                except IndexError:
                    break

    for o in range(1, maximum_order + 1):
        if clusters[o]:
            clusters[o] = cp.vstack(clusters[o]).get()
        else:
            print('Set of clusters of order {} is empty!'.format(o))
            clusters.pop(o)

    return clusters


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
        return np.array(list(edges), dtype=np.int32)

    nvertices = 1000
    max_degree = 5
    bonds = generate_graph(nvertices, max_degree)
    nbond = bonds.shape[0]
    logger.info(f"generate {nbond} bonds.\n")

    rows, cols = bonds.T
    data = np.ones_like(rows)
    max_vertex = np.max(bonds) + 1
    graph = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(max_vertex, max_vertex)).tocsr()

    # Create labels and n_components for the first function
    labels = np.zeros(nvertices, dtype=np.int32)
    n_components = 1

    b1 = benchmark(find_subclusters_cpu, (3, graph, labels, n_components, False), n_repeat=10)
    logger.info(b1)
    b2 = benchmark(find_subclusters_gpu, (3, graph, labels, n_components, False), n_repeat=10)
    logger.info(b2)

    logger.info(f"Achieving {int(np.mean(b1.cpu_times) / np.mean(b2.cpu_times))} speed up!")
