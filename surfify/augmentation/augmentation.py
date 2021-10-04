import numpy as np
from joblib import Memory
from scipy.spatial.transform import Rotation
from surfify.utils import neighbors, MeshProjector, recursively_find_neighbors


class SphericalAugmentation(object):
    """ Meta-class for spherical augmentations

    Attributes
    ----------
    vertices: array (N, 3)
        icosahedron's vertices
    triangles: array (M, 3)
        icosahdron's triangles
    neighbors: dict
        neighbors of each vertex
    """
    def __init__(self, cachedir, vertices, triangles, verbose=False):
        memory = Memory(cachedir, verbose=int(verbose))
        cached_neighbors = memory.cache(neighbors)
        self.vertices = vertices
        self.triangles = triangles
        self.neighbors = cached_neighbors(
            vertices, triangles, direct_neighbor=True)

    def __call__(self):
        raise NotImplementedError()


class SphericalRotation(SphericalAugmentation):
    """ Rotation of the icosahedron's vertices

    Attributes
    ----------
    rotation: scipy.spatial.transform.Rotation
        instance of a rotation
    rotated_vertices: array (N, 3)
        vertices rotated by the rotation operation
    projector: surfify.utils.MeshProjector
        projector to project back onto the original icosahedron
    """
    def __init__(self, cachedir, vertices, triangles, verbose=False,
                 angles=(5, 0, 0), compute_bary=False):
        super().__init__(cachedir, vertices, triangles, verbose)
        self.rotation = Rotation.from_euler('xyz', angles, degrees=True)
        self.rotated_vertices = self.rotation.apply(vertices)
        self.projector = MeshProjector(vertices, self.rotated_vertices,
                                       triangles, compute_bary, cachedir)

    def __call__(self, data, **kwargs):
        """ Rotates the provided texture and projects it onto the mesh,
        by considering it associated to the rotated mesh
        """
        return self.projector.project(data)


class SphericalRandomCut(SphericalAugmentation):
    """ Random cut of patches on the icosahedron

    Attributes
    ----------
    cut_size: int
        neighborhood order of the cut
    n_cut: int
        number of cuts
    cut_value: float
        value to replace to original values with on the icosahedron
    """
    def __init__(self, cachedir, vertices, triangles, verbose=False,
                 cut_size=3, n_cut=1, cut_value=0):
        super().__init__(cachedir, vertices, triangles, verbose)
        self.cut_size = cut_size
        self.n_cut = n_cut
        self.cut_value = cut_value

    def cut_out(self, data):
        """ Cuts out from data by randomly selecting the nodes to cut from
        and replacing the values
        """
        new_data = np.copy(data)
        indices_to_cut = []
        for i in range(self.n_cut):
            selected_node = int(np.random.rand()*len(self.vertices))
            indices_to_cut += recursively_find_neighbors(
                    start_node=selected_node, order=self.cut_size,
                    neighbors=self.neighbors)
        indices_to_cut = np.array(list(set(indices_to_cut)), dtype=int)
        new_data[indices_to_cut] = self.cut_value
        return new_data

    def __call__(self, data, **kwargs):
        """ Applies the cut out augmentation to the data
        """
        return self.cut_out(data)
