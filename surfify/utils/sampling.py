# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Spherical sampling & associated utilities.
"""

# Imports
import os
import collections
import numpy as np
from joblib import Memory
from math import sqrt, degrees
from sklearn.neighbors import BallTree
import networkx as nx
from nilearn.surface import load_surf_data, load_surf_mesh
import time


def interpolate(vertices, target_vertices, target_triangles):
    """ Interpolate missing data.

    Parameters
    ----------
    vertices: array (n_samples, n_dim)
        points of data set.
    target_vertices: array (n_query, n_dim)
        points to find interpolated texture for.
    target_triangles: array (n_query, 3)
        the mesh geometry definition.

    Returns
    -------
    interp_textures: array (n_query, n_feats)
        the interplatedd textures.
    """
    interp_textures = collections.OrderedDict()
    graph = vertex_adjacency_graph(target_vertices, target_triangles)
    common_vertices = downsample(target_vertices, vertices)
    missing_vertices = set(range(len(target_vertices))) - set(common_vertices)
    for node in sorted(graph.nodes):
        if node in common_vertices:
            interp_textures[node] = [node] * 2
        else:
            node_neighs = [idx for idx in graph.neighbors(node)
                           if idx in common_vertices]
            node_weights = np.linalg.norm(
                target_vertices[node_neighs] - target_vertices[node], axis=1)
            interp_textures[node] = node_neighs
    return interp_textures


def neighbors(vertices, triangles, depth=1, direct_neighbor=False):
    """ Build mesh vertices neighbors.

    Parameters
    ----------
    vertices: array (N, 3)
        the icosahedron vertices.
    triangles: array (N, 3)
        the icosahedron triangles.
    depth: int, default 1
        depth to stop the neighbors search, only paths of length <= depth are
        returned.
    direct_neighbor: bool, default False
        each spherical surface is composed of two types of vertices: 1) 12
        vertices with each having only 5 direct neighbors; and 2) the
        remaining vertices with each having 6 direct neighbors. For those
        vertices with 6 neighbors, DiNe assigns the index 1 to the center
        vertex and the indices 2–7 to its neighbors sequentially according
        to the angle between the vector of center vertex to neighboring vertex
        and the x-axis in the tangent plane. For the 12 vertices with only
        5 neighbors, DiNe assigns the indices both 1 and 2 to the center
        vertex, and indices 3–7 to the neighbors in the same way as those
        vertices with 6 neighbors.

    Returns
    --------
    neighs: dict
        a dictionary with vertices row index as keys and a dictionary of
        neighbors vertices row indexes organized by rungs as values.
    """
    graph = vertex_adjacency_graph(vertices, triangles)
    neighs = collections.OrderedDict()
    for node in sorted(graph.nodes):
        node_neighs = {}
        # node_neighs = [idx for idx in graph.neighbors(node)]
        for neigh, ring in nx.single_source_shortest_path_length(
                graph, node, cutoff=depth).items():
            if ring == 0:
                continue
            node_neighs.setdefault(ring, []).append(neigh)
        if direct_neighbor:
            _node_neighs = []
            if depth == 1:
                delta = np.pi / 4
            elif depth == 2:
                delta = np.pi / 8
            else:
                raise ValueError("Direct neighbors implemented only for "
                                 "depth <= 2.")
            for ring, ring_neighs in node_neighs.items():
                angles = np.asarray([
                    get_angle_with_xaxis(vertices[node], vertices[node], vec)
                    for vec in vertices[ring_neighs]])
                angles += delta
                angles = np.degrees(np.mod(angles, 2 * np.pi))
                ring_neighs = [x for _, x in sorted(
                    zip(angles, ring_neighs), key=lambda pair: pair[0])]
                if depth == 1 and ring == 1:
                    if len(ring_neighs) == 5:
                        ring_neighs.append(node)
                    elif len(ring_neighs) != 6:
                        raise ValueError("Mesh is not an icosahedron.")
                if depth == 2 and ring == 2:
                    ring_neighs = ring_neighs[1::2]
                    if len(_node_neighs) + len(ring_neighs) == 10:
                        ring_neighs.extend([node] * 2)
                    elif len(_node_neighs) + len(ring_neighs) == 11:
                        ring_neighs.append(node)
                    elif len(_node_neighs) + len(ring_neighs) != 12:
                        raise ValueError("Mesh is not an icosahedron.")
                _node_neighs.extend(ring_neighs)
            _node_neighs.append(node)
            node_neighs = _node_neighs
        neighs[node] = node_neighs

    return neighs


def vertex_adjacency_graph(vertices, triangles):
    """ Build a networkx graph representation of the vertices and
    their connections in the mesh.

    Parameters
    ----------
    vertices: array (N, 3)
        the icosahedron vertices.
    triangles: array (N, 3)
        the icosahedron triangles.

    Returns
    -------
    graph: networkx.Graph
        Graph representing vertices and edges between
        them where vertices are nodes and edges are edges

    Examples
    ----------
    This is useful for getting nearby vertices for a given vertex,
    potentially for some simple smoothing techniques.
    >>> graph = mesh.vertex_adjacency_graph
    >>> graph.neighbors(0)
    > [1, 3, 4]
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(vertices)))
    edges, edges_triangle = triangles_to_edges(triangles)
    edges_cache = []
    for idx1, idx2 in edges:
        smaller_index = min(idx1, idx2)
        greater_index = max(idx1, idx2)
        key = "{0}-{1}".format(smaller_index, greater_index)
        if key in edges_cache:
            continue
        edges_cache.append(key)
        graph.add_edge(smaller_index, greater_index)
    return graph


def get_angle_with_xaxis(center, normal, point):
    """ Project a point to the sphere tangent plane and compute the angle
    with the x-axis.

    Parameters
    ----------
    center: array (3, )
        a point in the plane.
    normal: array (3, )
        the normal to the plane.
    points: array (3, )
        the points to be projected.
    """
    # Assert is array
    center = np.asarray(center)
    normal = np.asarray(normal)
    point = np.asarray(point)

    # Project points to plane
    vector = point - center
    dist = np.dot(vector, normal)
    projection = point - normal * dist

    # Compute normal of the new projected x-axis and y-axis
    if center[0] != 0 or center[1] != 0:
        nx = np.cross(np.array([0, 0, 1]), center)
        ny = np.cross(center, nx)
    else:
        nx = np.array([1, 0, 0])
        ny = np.array([0, 1, 0])

    # Compute the angle between projected points and the x-axis
    vector = projection - center
    unit_vector = vector
    if np.linalg.norm(vector) != 0:
        unit_vector = unit_vector / np.linalg.norm(vector)
    unit_nx = nx / np.linalg.norm(nx)
    cos_theta = np.dot(unit_vector, unit_nx)
    if cos_theta > 1.:
        cos_theta = 1.
    elif cos_theta < -1.:
        cos_theta = -1.
    angle = np.arccos(cos_theta)
    if np.dot(unit_vector, ny) < 0:
        angle = 2 * np.pi - angle

    return angle


def triangles_to_edges(triangles, return_index=False):
    """ Given a list of triangles, return a list of edges.

    Parameters
    ----------
    triangles: array int (N, 3)
        Vertex indices representing triangles.

    Returns
    -------
    edges: array int (N * 3, 2)
        Vertex indices representing edges.
    triangles_index: array (N * 3, )
        Triangle indexes.
    """
    # Each triangles has three edges
    edges = triangles[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2))

    # Edges are in order of triangles due to reshape
    triangles_index = np.tile(
        np.arange(len(triangles)), (3, 1)).T.reshape(-1)

    return edges, triangles_index


def downsample(vertices, target_vertices):
    """ Downsample by finding nearest neighbors.

    Parameters
    ----------
    vertices: array (n_samples, n_dim)
        points of data set.
    target_vertices: array (n_query, n_dim)
        points to find nearest neighbors for.

    Returns
    -------
    nearest_idx: array (n_query, )
        index of nearest neighbor in target_vertices for every point in
        vertices.
    """
    if vertices.size == 0 or target_vertices.size == 0:
        return np.array([], int), np.array([])
    tree = BallTree(vertices, leaf_size=2)
    distances, nearest_idx = tree.query(
        target_vertices, return_distance=True, k=1)
    n_duplicates = len(nearest_idx) - len(np.unique(nearest_idx))
    if n_duplicates:
        raise RuntimeError("Could not downsample proprely, '{0}' duplicates "
                           "were found. Are you using an icosahedron "
                           "mesh?".format(n_duplicates))
    return nearest_idx.squeeze()


def downsample_data(data, down_indices, neighs, aggregation=None):
    """ Downsample data to smaller icosahedron

    Parameters
    ----------
    data: array (n_samples, n_vertices, n_features)
        data to be downsampled
    down_indces: list
        contains the downsample vertices indices in the upper order
        icosahedron, for each downsampling
    neighs: list
        contains the neighbors of each vertex of the upper order
        icosahedron, for each downsampling
    aggregation: str, default None
        aggregation strategy over the higher order neighborhoods: 'mean',
        'median, 'max', 'min', 'sum' or None

    Returns
    -------
    downsampled_data: array (n_samples, new_n_vertices, n_features)
        downsampled data
    """
    assert aggregation in ['mean', 'median', 'max', 'min', 'sum', None]
    if len(data.shape) < 3:
        data = data[np.newaxis, :, :]
    data = data.transpose((0, 2, 1))
    for i in range(len(down_indices)):
        if aggregation is not None:
            down_neigh_indices = neighs[i][down_indices[i]]
            n_vertices, neigh_size = down_neigh_indices.shape

            data = data[:, :, down_neigh_indices].reshape(
                    len(data), data.shape[1], n_vertices, neigh_size)
            data = getattr(data, aggregation)(-1)
        else:
            data = data[:, :, down_indices[i]]
    return data.transpose(0, 2, 1).squeeze()


def downsample_ico(coordinates, triangles, by=1, new_coordinates=None):
    """ Downsample an icosahedron to one with a smaller order

    Parameters
    ----------
    coordinates: array (N, 3)
        coordinates of the icosahedron to reduce
    triangles: array (N, 3)
        triangles of the icosahedron to reduce
    by: int, default 1
        number of orders to reduce the icosahedron by
    new_coordinates: list or None, default None
        list of the coordinates of each lower order icosahedron, of
        length by. If not provided the default is to take the first
        coordinates of the higher order icosahedron as new coordinates
        for the lower order one

    Returns
    -------
    new_coordinates: array (N, 3)
        vertices of the newly downsampled icosahedorn
    new_triangles: array (N, 3)
        triangles of the newly downsampled icosahedron
    """
    assert new_coordinates is None \
        or type(new_coordinates) is list and len(new_coordinates) == by

    for i in range(by):
        former_order = order_of_ico_from_vertices(len(coordinates))
        next_n_vertices = number_of_ico_vertices(former_order - 1)
        if new_coordinates is None:
            new_coordinate = coordinates[:next_n_vertices]
        else:
            new_coordinate = new_coordinates[i]

        new_triangles = []
        down_indices = downsample(coordinates, new_coordinate)
        old_neighbors = neighbors(coordinates, triangles, direct_neighbor=True)
        old_neighbors = np.array(list(old_neighbors.values()))
        for i, downer in enumerate(down_indices):
            for j, neigh in enumerate(old_neighbors[downer]):
                if neigh != downer:
                    next_neigh = old_neighbors[downer][
                        (j+1) % len(old_neighbors[downer])]
                    neighs = old_neighbors[neigh]
                    neighs1 = old_neighbors[next_neigh]
                    candidate = [i]
                    for neigh_order2 in neighs:
                        if neigh_order2 in down_indices and \
                           neigh_order2 != downer:
                            indicein4 = down_indices.tolist().index(
                                neigh_order2)
                            candidate.append(indicein4)
                            break
                    for neigh_order2 in neighs1:
                        if neigh_order2 in down_indices and \
                           neigh_order2 != downer:
                            indicein4 = down_indices.tolist().index(
                                neigh_order2)
                            candidate.append(indicein4)
                            break
                    if set(candidate) not in new_triangles and \
                       len(candidate) == 3:
                        new_triangles.append(set(candidate))
        new_triangles = np.array([list(tri) for tri in new_triangles])
        coordinates = new_coordinate
        triangles = new_triangles
    return new_coordinate, new_triangles


def neighbors_rec(vertices, triangles, size=5, zoom=5):
    """ Build rectangular grid neighbors and weights.

    Parameters
    ----------
    vertices: array (N, 3)
        the icosahedron vertices.
    triangles: array (N, 3)
        the icosahedron triangles.
    size: int, default 5
        the rectangular grid size.
    zoom: int, default 5
        scale factor applied on the unit sphere to control the neighborhood
        density.

    Returns
    --------
    neighs: array (N, size**2, 3)
        grid samples neighbors for each vertex.
    weights: array (N, size**2, 3)
        grid samples weights with neighbors for each vertex.
    grid_in_sphere: array (N, size**2, 3)
        zoomed rectangular grid on the sphere vertices.
    """
    grid_in_sphere = np.zeros((len(vertices), size**2, 3), dtype=float)
    neighs = np.zeros((len(vertices), size**2, 3), dtype=int)
    weights = np.zeros((len(vertices), size**2, 3), dtype=float)
    for idx1, node in enumerate(vertices):
        grid_in_sphere[idx1], _ = get_rectangular_projection(
            node, size=size, zoom=zoom)
        for idx2, point in enumerate(grid_in_sphere[idx1]):
            dist = np.linalg.norm(vertices - point, axis=1)
            ordered_neighs = np.argsort(dist)
            neighs[idx1, idx2] = ordered_neighs[:3]
            weights[idx1, idx2] = dist[neighs[idx1, idx2]]
    return neighs, weights, grid_in_sphere


def recursively_find_neighbors(start_node, order, neighbors):
    """ Recursively find neighbors from a starting node up to a certain order

    Parameters
    ----------
    start_node: int
        node to start from
    order: int
        order up to which to look for neighbors
    neighbors: dict
        neighbors for each node

    returns
    -------
    indices: list
        indices of the neighbors of order order or lower
    """
    indices = []
    if order <= 0:
        return [start_node]
    for neigh in neighbors[start_node]:
        if order == 1:
            indices.append(neigh)
        else:
            indices += recursively_find_neighbors(neigh, order-1, neighbors)

    return list(set(indices))


def get_rectangular_projection(node, size=5, zoom=5):
    """ Project rectangular grid in 2D sapce into 3D spherical space.

    Parameters
    ----------
    node: array (3, )
        a point in the sphere.
    size: int, default 5
        the rectangular grid size.
    zoom: int, default 5
        scale factor applied on the unit sphere to control the neighborhood
        density.

    Returns
    -------
    grid_in_sphere: array (size**2, 3)
        zoomed rectangular grid on the sphere.
    grid_in_tplane: array (size**2, 3)
        zoomed rectangular grid in the tangent space.
    """
    # Check kernel size
    if (size % 2) == 0:
        raise ValueError("An odd kernel size is expected.")
    midsize = size // 2

    # Compute normal of the new projected x-axis and y-axis
    node = node.copy() * zoom
    if node[0] != 0 or node[1] != 0:
        nx = np.cross(np.array([0, 0, 1]), node)
        ny = np.cross(node, nx)
    else:
        nx = np.array([1, 0, 0])
        ny = np.array([0, 1, 0])
    nx = nx / np.linalg.norm(nx)

    ny = ny / np.linalg.norm(ny)

    # Caculate the grid coordinate in tangent plane and project back on sphere
    grid_in_tplane = np.zeros((size ** 2, 3))
    grid_in_sphere = np.zeros((size ** 2, 3))
    corner = node - midsize * nx + midsize * ny
    for row in range(size):
        for column in range(size):
            point = corner - row * ny + column * nx
            grid_in_tplane[row * size + column, :] = point
            grid_in_sphere[row * size + column, :] = (
                point / np.linalg.norm(point) * zoom)

    return grid_in_sphere, grid_in_tplane


def icosahedron(order=3):
    """ Define an icosahedron mesh of any order.

    Parameters
    ----------
    order: int, default 3
        the icosahedron order.

    Returns
    -------
    vertices: array (N, 3)
        the icosahedron vertices.
    triangles: array (N, 3)
        the icosahedron triangles.
    """
    middle_point_cache = {}
    r = (1 + np.sqrt(5)) / 2
    vertices = [
        normalize([-1, r, 0]),
        normalize([1, r, 0]),
        normalize([-1, -r, 0]),
        normalize([1, -r, 0]),
        normalize([0, -1, r]),
        normalize([0, 1, r]),
        normalize([0, -1, -r]),
        normalize([0, 1, -r]),
        normalize([r, 0, -1]),
        normalize([r, 0, 1]),
        normalize([-r, 0, -1]),
        normalize([-r, 0, 1])]
    triangles = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1]]

    for idx in range(order):
        subdiv = []
        for tri in triangles:
            v1 = middle_point(tri[0], tri[1], vertices, middle_point_cache)
            v2 = middle_point(tri[1], tri[2], vertices, middle_point_cache)
            v3 = middle_point(tri[2], tri[0], vertices, middle_point_cache)
            subdiv.append([tri[0], v1, v3])
            subdiv.append([tri[1], v2, v1])
            subdiv.append([tri[2], v3, v2])
            subdiv.append([v1, v2, v3])
        triangles = subdiv

    return np.asarray(vertices), np.asarray(triangles)


def icosahedron_fs(hemi, order=7):
    """ Loads the freesurfer icosahedron mesh of any order for the right
    hemishpere. If the file associated to the order does not exist, it
    builds the icosahedron by downsampling the icosahedron with the lowest
    order that is of higher order than the one desired, and for which we
    have the file

    Parameters
    ----------
    hemi: string
        hemisphere
    order: int, default 7
        the icosahedron order.

    Returns
    -------
    vertices: array (N, 3)
        the icosahedron vertices.
    triangles: array (N, 3)
        the icosahedron triangles.
    """
    assert hemi in ["rh", "lh"]
    path = "/i2bm/local/freesurfer-6.0.0/subjects/fsaverage{}"\
        "/surf/{}.sphere".format('' if order == 7 else order, hemi)
    last_order = order
    while not os.path.isfile(path) and last_order < 7:
        last_order += 1
        path = path.replace(
            'fsaverage{}'.format('' if last_order-1 == 7 else last_order-1),
            'fsaverage{}'.format('' if last_order == 7 else last_order))
    vertices, triangles = load_surf_mesh(path)
    if last_order != order:
        vertices, triangles = downsample_ico(
            vertices, triangles, by=last_order-order)
    return vertices, triangles


def normalize(vertex):
    """ Return vertex coordinates fixed to the unit sphere.
    """
    x, y, z = vertex
    length = sqrt(x**2 + y**2 + z**2)
    return [idx / length for idx in (x, y, z)]


def middle_point(point_1, point_2, vertices, middle_point_cache):
    """ Find a middle point and project to the unit sphere.
    """
    # We check if we have already cut this edge first to avoid duplicated verts
    smaller_index = min(point_1, point_2)
    greater_index = max(point_1, point_2)
    key = "{0}-{1}".format(smaller_index, greater_index)
    if key in middle_point_cache:
        return middle_point_cache[key]

    # If it's not in cache, then we can cut it
    vert_1 = vertices[point_1]
    vert_2 = vertices[point_2]
    middle = [sum(elems) / 2. for elems in zip(vert_1, vert_2)]
    vertices.append(normalize(middle))
    index = len(vertices) - 1
    middle_point_cache[key] = index

    return index


def number_of_ico_vertices(order=3):
    """ Get the number of vertices of an icosahedron of specific order.

    Parameters
    ----------
    order: int, default 3
        the icosahedron order.

    Returns
    -------
    vertices: array (N, 3)
        the icosahedron vertices.
    triangles: array (N, 3)
        the icosahedron triangles.
    """
    return 10 * 4 ** order + 2


def order_of_ico_from_vertices(n_vertices):
    """ Get the order of the icosahedron from the number of vertices it has

    Parameters
    ----------
    n_vertices: int
        the number of vertices of an icosahedron.

    Returns
    -------
    order: int
        the order of the icosahedron
    """
    order = np.log((n_vertices-2) / 10) / np.log(4)

    if int(order) != order:
        raise ValueError(
            "This number of vertices does not correspond to those of a"
            "regular icosahedron")
    return int(order)


def order_triangles_vertices(vertices, triangles, clockwise_from_center=True):
    """ Order the triangles' vertices to be in a clockwise order when viewed
    from the center of the sphere described by the icosahedron

    Parameters
    ----------
    vertices: array (N, 3)
        the icosahedron's vertices
    triangles: array (M, 3)
        the icosahedron's triangles
    clockwise_from_center: bool
        True for clockwise, False for counter clockwise

    Returns
    -------
    reordered_triangles: array (N, 3)
        triangles reordered
    """
    reordered_triangles = triangles.copy()
    for i, triangle in enumerate(triangles):
        A, B, C = vertices[triangle]
        N = np.cross((B - A), (C - A))
        w = N @ A
        if (clockwise_from_center and w >= 0) or \
           (not clockwise_from_center and w <= 0):
            reordered_triangles[i] = triangle[[0, 2, 1]]
    return reordered_triangles


class MeshProjector:
    """ Class to project an icosahedral mesh onto another

    Attributes
    ----------
    mesh: array (N, 3)
        mesh to project onto
    template_mesh: array (N, 3)
        mesh to project
    triangles: array (M, 3)
        triangles associated to the meshes
    neighbors: dict
        neighbors of each vertex
    triangles_membership: dict
        triangles that each vertex belong to
    projections: array (N, 3)
        coordinates of the projections
    proj_membership: array(N,)
        index of the triangle in which each projection belong
    Bs: array (N, 3)
        barycentric coordinates of each projection in the corresponding
        triangle
    eps: float
        machine precision
    """
    def __init__(self, mesh, template_mesh, triangles, compute_bary=True,
                 cachedir=None):
        assert len(mesh) == len(template_mesh)
        cached_neighbors = neighbors
        if cachedir is not None:
            self.memory = Memory(cachedir, verbose=0)
            cached_neighbors = self.memory.cache(neighbors)
        self.mesh = mesh
        self.template_mesh = template_mesh
        self.triangles = triangles
        self.neighbors = cached_neighbors(template_mesh, triangles)
        self.neighbors = {
            key: neighs[1] for key, neighs in self.neighbors.items()}
        self.triangles_membership = {
            node: [idx for idx, tri in enumerate(triangles) if node in tri]
            for node in range(len(mesh))}
        self.projections = np.zeros((len(mesh), 3))
        self.proj_membership = np.empty(len(mesh), dtype=int)
        self.proj_membership[:] = -1
        self.Bs = np.zeros((len(mesh), 3))
        self.eps = np.finfo(np.float64).eps
        if compute_bary:
            cached_barycentric_coordinates = self.memory.cache(
                self.get_barycentric_coordinates)
            self.Bs, self.projections, self.proj_membership, = \
                cached_barycentric_coordinates()

    # def reccursively_find_membership(self, last_idx):
    #     used_to_be_nan = []

    #     for neigh in self.neighbors[last_idx]:
    #         if neigh == last_idx:
    #             continue
    #         elif self.proj_membership[neigh] == -1:
    #             found = False
    #             for tri, triangle in enumerate(self.triangles):
    #                 T = self.template_mesh[triangle]
    #                 B = np.linalg.solve(T.T, self.mesh[neigh])
    #                 eps = self.eps
    #                 if sum((B >= 0) | (np.abs(B) <= self.eps)) == 3:
    #                     found = True
    #                     self.projections[neigh] = B @ T
    #                     self.proj_membership[neigh] = tri
    #                     self.Bs[neigh] = B
    #                     used_to_be_nan.append(neigh)
    #                     break
    #             if not found:
    #                 print(":'(")
    #     for neigh in used_to_be_nan:
    #         self.reccursively_find_membership(neigh)

    def get_barycentric_coordinates(self):
        """ Compute the barycentric coordinates

        Parameters
        ----------

        Returns
        -------
        Bs: array (N, 3)
            barycentric coordinates of each projection in the corresponding
            triangle
        projections: array (N, 3)
            coordinates of the projections
        proj_membership: array(N,)
            index of the triangle in which each projection belong
        """
        self.triangles = order_triangles_vertices(
            self.template_mesh, self.triangles)
        for node in range(len(self.mesh)):
            found = False
            for i, triangle in enumerate(self.triangles):
                T = self.template_mesh[triangle]
                B = np.linalg.solve(T.T, self.mesh[node])
                if sum((B >= 0) | (np.abs(B) <= self.eps)) == 3:
                    found = True
                    self.projections[node] = B @ T
                    self.proj_membership[node] = i
                    self.Bs[node] = B
                    break
            if not found:
                print(":'(")
        return self.Bs, self.projections, self.proj_membership

    def project(self, texture):
        """ Project a texture onto the icosahedron

        Parameters
        ----------
        texture: array (N, k)
            texture associated to the template mesh

        Returns
        -------
        new_texture: array (N, k)
            projected texture on the mesh
        """
        if (self.proj_membership == -1).sum() > 0:
            print((self.proj_membership == -1).sum())
            raise AttributeError("You need to compute the barycentric "
                                 "coordinates before projecting a texture")
        new_texture = np.zeros(texture.shape)

        for i in range(len(self.mesh)):
            triangle = self.triangles[self.proj_membership[i]]
            new_texture[i] = texture[triangle].T @ self.Bs[i]
        return new_texture
