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
import tempfile
import collections
import numpy as np
import networkx as nx
from scipy.spatial import transform
from sklearn.neighbors import BallTree
from nilearn.surface import load_surf_mesh
from nilearn.datasets import fetch_surf_fsaverage
from .io import HidePrints


def normalize(vertex):
    """ Return vertex coordinates fixed to the unit sphere.
    """
    x, y, z = vertex
    length = np.sqrt(x**2 + y**2 + z**2)
    return [idx / length for idx in (x, y, z)]


R = (1 + np.sqrt(5)) / 2
STANDARD_ICO = {
    "vertices": [
        normalize([-1, R, 0]),
        normalize([1, R, 0]),
        normalize([-1, -R, 0]),
        normalize([1, -R, 0]),
        normalize([0, -1, R]),
        normalize([0, 1, R]),
        normalize([0, -1, -R]),
        normalize([0, 1, -R]),
        normalize([R, 0, -1]),
        normalize([R, 0, 1]),
        normalize([-R, 0, -1]),
        normalize([-R, 0, 1])],
    "triangles": [
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
}


def neighbors(vertices, triangles, depth=1, direct_neighbor=False):
    """ Build mesh vertices neighbors.

    This is the base function to build Direct Neighbors (DiNe) kernels.

    See Also
    --------
    neighbors_rec

    Examples
    --------
    >>> from surfify.utils import icosahedron, neighbors
    >>> import matplotlib.pyplot as plt
    >>> from surfify.plotting import plot_trisurf
    >>> ico2_verts, ico2_tris = icosahedron(order=2)
    >>> neighs = neighbors(ico2_verts, ico2_tris, direct_neighbor=True)
    >>> fig, ax = plt.subplots(1, 1, subplot_kw={
            "projection": "3d", "aspect": "auto"}, figsize=(10, 10))
    >>> plot_trisurf(ico2_verts, triangles=ico2_tris, colorbar=False, fig=fig,
                     ax=ax)
    >>> center = ico2_verts[0]
    >>> for cnt, idx in enumerate(neighs[0]):
    >>>     point = ico2_verts[idx]
    >>>     ax.scatter(point[0], point[1], point[2], marker="o", c="red",
                       s=100)
    >>> ax.scatter(center[0], center[1], center[2], marker="o", c="blue",
                   s=100)
    >>> plt.show()

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
        vertex and the indices 2-7 to its neighbors sequentially according
        to the angle between the vector of center vertex to neighboring vertex
        and the x-axis in the tangent plane. For the 12 vertices with only
        5 neighbors, DiNe assigns the indices both 1 and 2 to the center
        vertex, and indices 3-7 to the neighbors in the same way as those
        vertices with 6 neighbors.

    Returns
    --------
    neighs: dict
        a dictionary with vertices row index as keys and a dictionary of
        neighbors vertices row indexes organized by rings as values.
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

    Examples
    --------
    This is useful for getting nearby vertices for a given vertex,
    potentially for some simple smoothing techniques.
    >>> graph = mesh.vertex_adjacency_graph
    >>> graph.neighbors(0)
    > [1, 3, 4]

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


# TODO: normalize weights (see rotate_data).
def neighbors_rec(vertices, triangles, size=5, zoom=5):
    """ Build rectangular grid neighbors and weights.

    This is the base function to build Rectangular Patch (RePa) kernels.

    See Also
    --------
    neighbors

    Examples
    --------
    >>> from surfify.utils import icosahedron, neighbors_rec
    >>> import matplotlib.pyplot as plt
    >>> from surfify.plotting import plot_trisurf
    >>> ico2_verts, ico2_tris = icosahedron(order=2)
    >>> neighs = neighbors_rec(ico2_verts, ico2_tris, size=3, zoom=3)
    >>> fig, ax = plt.subplots(1, 1, subplot_kw={
            "projection": "3d", "aspect": "auto"}, figsize=(10, 10))
    >>> plot_trisurf(ico2_verts, triangles=ico2_tris, colorbar=False, fig=fig,
                     ax=ax)
    >>> center = ico2_verts[0]
    >>> for cnt, point in enumerate(neighs[2][0]):
    >>>     ax.scatter(point[0], point[1], point[2], marker="o", c="red",
                       s=100)
    >>> ax.scatter(center[0], center[1], center[2], marker="o", c="blue",
                   s=100)
    >>> plt.show()

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


def get_rectangular_projection(node, size=5, zoom=5):
    """ Project 2D rectangular grid defined in node tangent space into 3D
    spherical space.

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
    node = node.copy()
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
    spacing = 1 / zoom
    midsize *= spacing
    corner = node - midsize * nx + midsize * ny
    for row in range(size):
        for column in range(size):
            point = corner - row * spacing * ny + column * spacing * nx
            grid_in_tplane[row * size + column, :] = point
            grid_in_sphere[row * size + column, :] = (
                point / np.linalg.norm(point))

    return grid_in_sphere, grid_in_tplane


def find_neighbors(start_node, order, neighbors):
    """ Recursively find neighbors from a starting node up to a certain order.

    See Also
    --------
    neighbors, neighbors_rec

    Examples
    --------
    >>> from surfify.utils import icosahedron, neighbors_rec, find_neighbors
    >>> import matplotlib.pyplot as plt
    >>> from surfify.plotting import plot_trisurf
    >>> ico2_verts, ico2_tris = icosahedron(order=2)
    >>> neighs = neighbors_rec(ico2_verts, ico2_tris, size=3, zoom=3)[0]
    >>> neighs = neighs.reshape(len(neighs), -1)
    >>> neighs = neighbors(ico2_verts, ico2_tris, depth=1,
                           direct_neighbor=True)
    >>> node = 0
    >>> node_neighs = find_neighbors(node, order=3, neighbors=neighs)
    >>> fig, ax = plt.subplots(1, 1, subplot_kw={
            "projection": "3d", "aspect": "auto"}, figsize=(10, 10))
    >>> plot_trisurf(ico2_verts, triangles=ico2_tris, colorbar=False, fig=fig,
                     ax=ax)
    >>> center = ico2_verts[node]
    >>> for cnt, idx in enumerate(node_neighs):
    >>>     point = ico2_verts[idx]
    >>>     ax.scatter(point[0], point[1], point[2], marker="o", c="red",
                       s=100)
    >>> ax.scatter(center[0], center[1], center[2], marker="o", c="blue",
                   s=100)
    >>> plt.show()

    Parameters
    ----------
    start_node: int
        node index to start search from.
    order: int
        order up to which to look for neighbors.
    neighbors: dict
        neighbors for each node as generated by the 'neighbors' or
        'neighbors_rec' functions.

    returns
    -------
    indices: list of int
        the n-ring neighbors indices.
    """
    indices = []
    if order <= 0:
        return [start_node]
    for neigh in neighbors[start_node]:
        if order == 1:
            indices.append(neigh)
        else:
            indices += find_neighbors(neigh, order - 1, neighbors)
    return list(set(indices))


def build_freesurfer_ico(ico_file):
    """ Build FreeSurfer reference icosahedron by fetching existing data
    and building lower orders using downsampling.

    Freesurfer coordinates are between -100 and 100, and are rescaled between
    -1 and 1.

    Parameters
    ----------
    ico_file: str
        path to the generated FreeSurfer reference icosahedron topologies.
    """
    data = {}
    for order in range(7, 2, -1):
        surf_name = "fsaverage{0}".format(order)
        with HidePrints(hide_err=True):
            with tempfile.TemporaryDirectory() as tmpdir:
                fsaverage = fetch_surf_fsaverage(
                    mesh=surf_name, data_dir=tmpdir)
                vertices, triangles = load_surf_mesh(fsaverage["sphere_left"])
            vertices /= 100.
            data[surf_name + ".vertices"] = vertices.astype(np.float32)
            data[surf_name + ".triangles"] = triangles
    for order in range(2, -1, -1):
        surf_name = "fsaverage{0}".format(order)
        up_vertices = data["fsaverage{0}.vertices".format(order + 1)]
        up_triangles = data["fsaverage{0}.triangles".format(order + 1)]
        vertices, triangles = downsample_ico(up_vertices, up_triangles, by=1)
        data[surf_name + ".vertices"] = vertices
        data[surf_name + ".triangles"] = triangles
    np.savez(ico_file, **data)


def icosahedron(order=3, standard_ico=False):
    """ Define an icosahedron mesh of any order.

    Examples
    --------
    >>> from surfify.utils import icosahedron
    >>> import matplotlib.pyplot as plt
    >>> from surfify.plotting import plot_trisurf
    >>> ico3_verts, ico3_tris = icosahedron(order=3)
    >>> print(ico3_verts.shape, ico3_tris.shape)
    >>> fig, ax = plt.subplots(1, 1, subplot_kw={
            "projection": "3d", "aspect": "auto"}, figsize=(10, 10))
    >>> plot_trisurf(ico3_verts, triangles=ico3_tris, colorbar=False, fig=fig,
                     ax=ax)
    >>> plt.show()

    Parameters
    ----------
    order: int, default 3
        the icosahedron order.
    standard_ico: bool, default False
        optionally uses a standard icosahedron tessalation. FreeSurfer
        tesselation is used by default.

    Returns
    -------
    vertices: array (N, 3)
        the icosahedron vertices.
    triangles: array (M, 3)
        the icosahedron triangles.
    """
    if standard_ico:
        vertices = STANDARD_ICO["vertices"].copy()
        triangles = STANDARD_ICO["triangles"].copy()
        middle_point_cache = {}
        for _ in range(order):
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
        vertices = np.asarray(vertices)
        triangles = np.asarray(triangles)
    else:
        resource_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "resources")
        resource_file = os.path.join(resource_dir, "freesurfer_icos.npz")
        icos = np.load(resource_file)
        surf_name = "fsaverage{0}".format(order)
        try:
            vertices = icos[surf_name + ".vertices"]
            triangles = icos[surf_name + ".triangles"]
        except Exception as err:
            print("-- available topologies:", icos.files)
            raise(err)

    return vertices, triangles


def middle_point(point_1, point_2, vertices, middle_point_cache):
    """ Find a middle point and project it to the unit sphere.

    This function is only used to build an icosahedron geometry.
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

    See Also
    --------
    order_of_ico_from_vertices

    Examples
    --------
    >>> from surfify.utils import number_of_ico_vertices, icosahedron
    >>> ico3_verts, ico3_tris = icosahedron(order=3)
    >>> n_verts = number_of_ico_vertices(order=3)
    >>> print(n_verts, ico3_verts.shape)

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
    """ Get the order of an icosahedron from his number of vertices.

    See Also
    --------
    number_of_ico_vertices

    Examples
    --------
    >>> from surfify.utils import order_of_ico_from_vertices, icosahedron
    >>> ico3_verts, ico3_tris = icosahedron(order=3)
    >>> order = order_of_ico_from_vertices(len(ico3_verts))
    >>> print(order)

    Parameters
    ----------
    n_vertices: int
        the number of vertices of an icosahedron.

    Returns
    -------
    order: int
        the order of the icosahedron
    """
    order = np.log((n_vertices - 2) / 10) / np.log(4)
    if int(order) != order:
        raise ValueError(
            "This number of vertices does not correspond to those of a "
            "regular icosahedron.")
    return int(order)


def interpolate(vertices, target_vertices, target_triangles):
    """ Interpolate icosahedron missing data by finding nearest neighbors.

    Interpolation weights are set to 1 for a regular icosahedron geometry.

    See Also
    --------
    interpolate_data, downsample, downsample_data, downsample_ico

    Examples
    --------
    >>> from surfify.utils import icosahedron, interpolate
    >>> from surfify.datasets import make_classification
    >>> import matplotlib.pyplot as plt
    >>> from surfify.plotting import plot_trisurf
    >>> ico2_verts, ico2_tris = icosahedron(order=2)
    >>> ico3_verts, ico3_tris = icosahedron(order=3)
    >>> X, y = make_classification(ico2_verts, n_samples=1, n_classes=3,
                                   scale=1, seed=42)
    >>> up_indices = interpolate(ico2_verts, ico3_verts, ico3_tris)
    >>> up_indices = np.asarray(list(up_indices.values()))
    >>> y_up = y[up_indices.reshape(-1)].reshape(up_indices.shape)
    >>> y_up = np.mean(y_up, axis=-1)
    >>> plot_trisurf(ico3_verts, triangles=ico3_tris, texture=y_up,
                     is_label=False)
    >>> plt.show()

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
    interp_indices: array (n_query, n_feats)
        the interpolation indices.
    """
    interp_indices = collections.OrderedDict()
    interp_weights = collections.OrderedDict()
    graph = vertex_adjacency_graph(target_vertices, target_triangles)
    common_vertices = downsample(target_vertices, vertices)
    missing_vertices = set(range(len(target_vertices))) - set(common_vertices)
    for node in sorted(graph.nodes):
        if node in common_vertices:
            interp_indices[node] = [node] * 2
            interp_weights[node] = [1., 1.]
        else:
            node_neighs = [idx for idx in graph.neighbors(node)
                           if idx in common_vertices]
            node_weights = np.linalg.norm(
                target_vertices[node_neighs] - target_vertices[node], axis=1)
            interp_indices[node] = node_neighs
            interp_weights[node] = node_weights
    return interp_indices


def interpolate_data(data, by=1, up_indices=None):
    """ Interpolate data/texture on the icosahedron to an upper order.

    See Also
    --------
    interpolate, downsample, downsample_data, downsample_ico

    Examples
    --------
    >>> from surfify.utils import icosahedron, interpolate_data
    >>> from surfify.datasets import make_classification
    >>> import matplotlib.pyplot as plt
    >>> from surfify.plotting import plot_trisurf
    >>> ico2_verts, ico2_tris = icosahedron(order=2)
    >>> ico4_verts, ico4_tris = icosahedron(order=4)
    >>> X, y = make_classification(ico2_verts, n_samples=1, n_classes=3,
                                   scale=1, seed=42)
    >>> y = y.reshape(1, -1, 1)
    >>> y_up = interpolate_data(y, by=2).squeeze()
    >>> plot_trisurf(ico4_verts, triangles=ico4_tris, texture=y_up,
                     is_label=False)
    >>> plt.show()

    Parameters
    ----------
    data: array (n_samples, n_vertices, n_features)
        data to be upsampled.
    by: int, default 1
        number of orders to increase the icosahedron by.
    up_indices: list of array, default None
        optionally specify the list of consecutive upsampling vertices
        indices.

    Returns
    -------
    upsampled_data: array (n_samples, new_n_vertices, n_features)
        upsampled data.
    """
    if len(data.shape) != 3:
        raise ValueError(
            "Unexpected input data. Must be (n_samples, n_vertices, "
            "n_features) but got '{0}'.".format(data.shape))
    if up_indices is None:
        order = order_of_ico_from_vertices(data.shape[1])
        ico_verts, _ = icosahedron(order)
        up_indices = []
        for up_order in range(order + 1, order + 1 + by, 1):
            up_ico_verts, up_ico_tris = icosahedron(up_order)
            _up_indices = interpolate(ico_verts, up_ico_verts, up_ico_tris)
            up_indices.append(np.asarray(list(_up_indices.values())))
            ico_verts = up_ico_verts
    n_samples = len(data)
    n_features = data.shape[-1]
    for indices in up_indices:
        n_new_vertices, n_neighs = indices.shape
        data = data[:, indices.reshape(-1)].reshape(
            n_samples, n_new_vertices, n_neighs, n_features)
        data = np.mean(data, axis=2)
    return data


def downsample(vertices, target_vertices):
    """ Downsample icosahedron vertices by finding nearest neighbors.

    See Also
    --------
    downsample_data, downsample_ico, interpolate, interpolate_data

    Examples
    --------
    >>> from surfify.utils import icosahedron, downsample
    >>> import matplotlib.pyplot as plt
    >>> from surfify.plotting import plot_trisurf
    >>> ico2_verts, ico2_tris = icosahedron(order=2)
    >>> ico3_verts, ico3_tris = icosahedron(order=3)
    >>> down3to2 = downsample(ico3_verts, ico2_verts)
    >>> ico3_down_vertices = ico3_verts[down3to2]
    >>> fig, ax = plt.subplots(1, 1, subplot_kw={
            "projection": "3d", "aspect": "auto"}, figsize=(10, 10))
    >>> plot_trisurf(ico3_verts, triangles=ico3_tris, colorbar=False, fig=fig,
                     ax=ax)
    >>> for cnt, point in enumerate(ico3_down_vertices):
    >>>     ax.scatter(point[0], point[1], point[2], marker="o", c="red",
                       s=100)
    >>> plt.show()

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


def downsample_data(data, by=1, down_indices=None):
    """ Downsample data/texture on the icosahedron to a lower order.

    See Also
    --------
    downsample, downsample_ico, interpolate, interpolate_data

    Examples
    --------
    >>> from surfify.utils import icosahedron, downsample_data
    >>> from surfify.datasets import make_classification
    >>> import matplotlib.pyplot as plt
    >>> from surfify.plotting import plot_trisurf
    >>> ico2_verts, ico2_tris = icosahedron(order=2)
    >>> ico4_verts, ico4_tris = icosahedron(order=4)
    >>> X, y = make_classification(ico4_verts, n_samples=1, n_classes=3,
                                   scale=1, seed=42)
    >>> y = y.reshape(1, -1, 1)
    >>> y_down = downsample_data(y, by=2).squeeze()
    >>> plot_trisurf(ico2_verts, triangles=ico2_tris, texture=y_down,
                     is_label=True)
    >>> plt.show()

    Parameters
    ----------
    data: array (n_samples, n_vertices, n_features)
        data to be downsampled.
    by: int, default 1
        number of orders to reduce the icosahedron by.
    down_indices: list of array, default None
        optionally specify the list of consecutive downsampling vertices
        indices.

    Returns
    -------
    downsampled_data: array (n_samples, new_n_vertices, n_features)
        downsampled data.
    """
    if len(data.shape) != 3:
        raise ValueError(
            "Unexpected input data. Must be (n_samples, n_vertices, "
            "n_features) but got '{0}'.".format(data.shape))
    if down_indices is None:
        order = order_of_ico_from_vertices(data.shape[1])
        ico_verts, _ = icosahedron(order)
        down_indices = []
        for low_order in range(order - 1, order - 1 - by, -1):
            low_ico_verts, _ = icosahedron(low_order)
            down_indices.append(downsample(ico_verts, low_ico_verts))
            ico_verts = low_ico_verts
    for indices in down_indices:
        data = data[:, indices]
    return data


# TODO: add a description of the method.
def downsample_ico(vertices, triangles, by=1, down_indices=None):
    """ Downsample an icosahedron full geometry: vertices and triangles.

    See Also
    --------
    downsample, downsample_data, interpolate, interpolate_data

    Examples
    --------
    >>> from surfify.utils import icosahedron, downsample_ico
    >>> import matplotlib.pyplot as plt
    >>> from surfify.plotting import plot_trisurf
    >>> ico4_verts, ico4_tris = icosahedron(order=4)
    >>> ico2_down_verts, ico2_down_tris = downsample_ico(
            ico4_verts, ico4_tris, by=2)
    >>> plot_trisurf(ico2_down_verts, triangles=ico2_down_tris, colorbar=False)
    >>> plt.show()

    Parameters
    ----------
    vertices: array (N, 3)
        vertices of the icosahedron to reduce.
    triangles: array (N, 3)
        triangles of the icosahedron to reduce.
    by: int, default 1
        number of orders to reduce the icosahedron by.
    down_indices: list of array, default None
        optionally specify the list of consecutive downsampling vertices
        indices.

    Returns
    -------
    downsampled_vertices: array (M, 3)
        vertices of the newly downsampled icosahedorn.
    downsampled_triangles: array (M, 3)
        triangles of the newly downsampled icosahedron.
    """
    for idx_order in range(by):

        former_order = order_of_ico_from_vertices(len(vertices))
        if down_indices is None:
            new_vertices, _ = icosahedron(former_order - 1)
            indices = downsample(vertices, new_vertices)
        else:
            indices = down_indices[idx_order]
            new_vertices = vertices[indices]

        new_triangles = []
        former_neighbors = neighbors(vertices, triangles, direct_neighbor=True)
        former_neighbors = np.array(list(former_neighbors.values()))
        for idx_down, down_node in enumerate(indices):
            for idx_neigh, neigh_node in enumerate(
                    former_neighbors[down_node]):
                if neigh_node != down_node:
                    next_neigh_node = former_neighbors[down_node][
                        (idx_neigh + 1) % len(former_neighbors[down_node])]
                    neigh_node_neighs = former_neighbors[neigh_node]
                    next_neigh_node_neighs = former_neighbors[next_neigh_node]
                    candidates = [idx_down]
                    for neighs in (neigh_node_neighs, next_neigh_node_neighs):
                        for neigh_idx in neighs:
                            if neigh_idx in indices and neigh_idx != down_node:
                                candidates.append(
                                    indices.tolist().index(neigh_idx))
                                break
                    if (set(candidates) not in new_triangles and
                            len(candidates) == 3):
                        new_triangles.append(set(candidates))

        new_triangles = np.array([list(tri) for tri in new_triangles])
        vertices = new_vertices
        triangles = new_triangles

    return new_vertices, new_triangles


def rotate_data(data, vertices, triangles, angles):
    """ Rotate data/texture on an icosahedron.

    Examples
    --------
    >>> from surfify.utils import icosahedron, rotate_data
    >>> from surfify.datasets import make_classification
    >>> import matplotlib.pyplot as plt
    >>> from surfify.plotting import plot_trisurf
    >>> ico3_verts, ico3_tris = icosahedron(order=3)
    >>> X, y = make_classification(ico3_verts, n_samples=1, n_classes=3,
                                   scale=1, seed=42)
    >>> y_rot = rotate_data(y.reshape(1, -1, 1), ico3_verts, ico3_tris,
                            (45, 0, 0)).squeeze()
    >>> plot_trisurf(ico3_verts, triangles=ico3_tris, texture=y,
                     is_label=False)
    >>> plot_trisurf(ico3_verts, triangles=ico3_tris, texture=y_rot,
                     is_label=False)
    >>> plt.show()

    Parameters
    ----------
    data: array (n_samples, n_vertices, n_features)
        data to be rotated.
    vertices: array (N, 3)
        vertices of the icosahedron.
    triangles: array (N, 3)
        triangles of the icosahedron.
    angles: 3-uplet
        the rotation angles in degrees for each axis (Euler representation).

    Returns
    -------
    rotated_data: array (n_samples, n_vertices, n_features)
        rotated data.
    """
    if len(data.shape) != 3:
        raise ValueError(
            "Unexpected input data. Must be (n_samples, n_vertices, "
            "n_features) but got '{0}'.".format(data.shape))

    rotation = transform.Rotation.from_euler("xyz", angles, degrees=True)
    rotated_vertices = rotation.apply(vertices)

    neighs = np.zeros((len(vertices), 3), dtype=int)
    weights = np.zeros((len(vertices), 3), dtype=float)
    for idx, point in enumerate(rotated_vertices):
        dist = np.linalg.norm(vertices - point, axis=1)
        ordered_neighs = np.argsort(dist)
        neighs[idx] = ordered_neighs[:3]
        weights[idx] = dist[neighs[idx]] / np.sum(dist[neighs[idx]])
    n_samples = len(data)
    n_features = data.shape[-1]
    n_vertices, n_neighs = neighs.shape
    flat_neighs = neighs.reshape(-1)
    flat_weights = np.repeat(weights.reshape(1, -1, 1), n_samples, axis=0)
    rotated_data = data[:, flat_neighs] * flat_weights
    rotated_data = rotated_data.reshape(n_samples, n_vertices, n_neighs,
                                        n_features)
    rotated_data = np.sum(rotated_data, axis=2)

    return rotated_data


def order_triangles(vertices, triangles, clockwise_from_center=True):
    """ Order the icosahedron triangles to be in a clockwise order when viewed
    from the center of the sphere.

    Examples
    --------
    >>> from surfify.utils import icosahedron, order_triangles
    >>> ico0_verts, ico0_tris = icosahedron(order=0)
    >>> clockwise_ico0_tris = order_triangles(
            ico0_verts, ico0_tris, clockwise_from_center=True)
    >>> counter_clockwise_ico0_tris = order_triangles(
            ico0_verts, ico0_tris, clockwise_from_center=False)
    >>> print(clockwise_ico0_tris)
    >>> print(counter_clockwise_ico0_tris)

    Parameters
    ----------
    vertices: array (N, 3)
        the icosahedron's vertices.
    triangles: array (M, 3)
        the icosahedron's triangles.
    clockwise_from_center: bool, default True
        optionally use counter clockwise order.

    Returns
    -------
    reordered_triangles: array (M, 3)
        reordered triangles.
    """
    reordered_triangles = triangles.copy()
    for idx, triangle in enumerate(triangles):
        loc_x, loc_y, loc_z = vertices[triangle]
        norm = np.cross((loc_y - loc_x), (loc_z - loc_x))
        w = np.dot(norm, loc_x)
        if ((clockwise_from_center and w >= 0) or
                (not clockwise_from_center and w <= 0)):
            reordered_triangles[idx] = triangle[[0, 2, 1]]
    return reordered_triangles
