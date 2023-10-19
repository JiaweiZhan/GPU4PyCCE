import unittest
from pycce.find_clusters import find_subclusters
from pycce.cuda_code.cluster.find_subclusters import find_subclusters_gpu
import numpy as np
import scipy

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

class TestFindSubclusters(unittest.TestCase):

    def test_acc_strong_false(self):
        nvertices = 1000
        max_degree = 5
        bonds = generate_graph(nvertices, max_degree)
        nbond = bonds.shape[0]
        print(f"generate {nbond} bonds.")

        rows, cols = bonds.T
        data = np.ones_like(rows)
        max_vertex = np.max(bonds) + 1
        graph = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(max_vertex, max_vertex)).tocsr()

        # Create labels and n_components for the first function
        labels = np.zeros(nvertices, dtype=np.int32)
        n_components = 1

        # Call the first function
        result1 = find_subclusters(3, graph, labels, n_components, False)[3]
        result2 = find_subclusters_gpu(bonds, strong=False).get()
        self.assertTrue(np.allclose(result1, result2))

    def test_acc_strong_true(self):
        nvertices = 1000
        max_degree = 5
        bonds = generate_graph(nvertices, max_degree)
        nbond = bonds.shape[0]
        print(f"generate {nbond} bonds.")

        rows, cols = bonds.T
        data = np.ones_like(rows)
        max_vertex = np.max(bonds) + 1
        graph = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(max_vertex, max_vertex)).tocsr()

        # Create labels and n_components for the first function
        labels = np.zeros(nvertices, dtype=np.int32)
        n_components = 1

        # Call the first function
        result1 = find_subclusters(3, graph, labels, n_components, True)[3]
        result2 = find_subclusters_gpu(bonds, strong=True).get()
        self.assertTrue(np.allclose(result1, result2))

if __name__ == '__main__':
    unittest.main()
