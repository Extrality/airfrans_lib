import numpy as np

def cell_sampling_2d(cell_points, cell_attr = None):
    '''
    Sample points in a two dimensional cell via parallelogram sampling and triangle interpolation via barycentric coordinates.
    The vertices have to be ordered in a certain way.

    Args:
        cell_points (np.ndarray): Vertices of the 2 dimensional cells. Shape `(N, 4)` for N cells with 4 vertices.
        cell_attr (np.ndarray, optional): Features of the vertices of the 2 dimensional cells. Shape `(N, 4, k)` for N cells with 4 edges and k features. 
            If given shape `(N, 4)` it will resize it automatically in a `(N, 4, 1)` tensor. Default: ``None``
    '''
    # Sampling via triangulation of the cell and parallelogram sampling
    v0, v1 = cell_points[:, 1] - cell_points[:, 0], cell_points[:, 3] - cell_points[:, 0]
    v2, v3 = cell_points[:, 3] - cell_points[:, 2], cell_points[:, 1] - cell_points[:, 2]  
    a0, a1 = np.abs(np.linalg.det(np.hstack([v0[:, :2], v1[:, :2]]).reshape(-1, 2, 2))), \
                np.abs(np.linalg.det(np.hstack([v2[:, :2], v3[:, :2]]).reshape(-1, 2, 2)))
    p = a0/(a0 + a1)
    index_triangle = np.random.binomial(1, p)[:, None]
    u = np.random.uniform(size = (len(p), 2))
    sampled_point = index_triangle*(u[:, 0:1]*v0 + u[:, 1:2]*v1) + (1 - index_triangle)*(u[:, 0:1]*v2 + u[:, 1:2]*v3)
    sampled_point_mirror = index_triangle*((1 - u[:, 0:1])*v0 + (1 - u[:, 1:2])*v1) + (1 - index_triangle)*((1 - u[:, 0:1])*v2 + (1 - u[:, 1:2])*v3)
    reflex = (u.sum(axis = 1) > 1)
    sampled_point[reflex] = sampled_point_mirror[reflex]

    # Interpolation on a triangle via barycentric coordinates
    if cell_attr is not None:
        t0, t1, t2 = np.zeros_like(v0), index_triangle*v0 + (1 - index_triangle)*v2, index_triangle*v1 + (1 - index_triangle)*v3
        w = (t1[:, 1] - t2[:, 1])*(t0[:, 0] - t2[:, 0]) + (t2[:, 0] - t1[:, 0])*(t0[:, 1] - t2[:, 1])
        w0 = (t1[:, 1] - t2[:, 1])*(sampled_point[:, 0] - t2[:, 0]) + (t2[:, 0] - t1[:, 0])*(sampled_point[:, 1] - t2[:, 1])
        w1 = (t2[:, 1] - t0[:, 1])*(sampled_point[:, 0] - t2[:, 0]) + (t0[:, 0] - t2[:, 0])*(sampled_point[:, 1] - t2[:, 1])
        w0, w1 = w0/w, w1/w
        w2 = 1 - w0 - w1
        
        if len(cell_attr.shape) == 2:
            cell_attr = cell_attr[:, :, None]
        attr0 = index_triangle*cell_attr[:, 0] + (1 - index_triangle)*cell_attr[:, 2]
        attr1 = index_triangle*cell_attr[:, 1] + (1 - index_triangle)*cell_attr[:, 1]
        attr2 = index_triangle*cell_attr[:, 3] + (1 - index_triangle)*cell_attr[:, 3]
        sampled_attr = w0[:, None]*attr0 + w1[:, None]*attr1 + w2[:, None]*attr2

    sampled_point += index_triangle*cell_points[:, 0] + (1 - index_triangle)*cell_points[:, 2]    

    return np.hstack([sampled_point[:, :2], sampled_attr]) if cell_attr is not None else sampled_point[:, :2]

def cell_sampling_1d(line_points, line_attr = None):
    '''
    Sample points in a one dimensional cell via linear sampling and interpolation.

    Args:
        line_points (np.ndarray): Edges of the 1 dimensional cells. Shape `(N, 2)` for N cells with 2 edges.
        line_attr (np.ndarray, optional): Features of the edges of the 1 dimensional cells. Shape `(N, 2, k)` for N cells with 2 edges and k features.
            If given shape `(N, 2)` it will resize it automatically in a `(N, 2, 1)` tensor. Default: ``None``
    '''
    # Linear sampling
    u = np.random.uniform(size = (len(line_points), 1))
    sampled_point = u*line_points[:, 0] + (1 - u)*line_points[:, 1]

    # Linear interpolation
    if line_attr is not None:   
        if len(line_attr.shape) == 2:
            line_attr = line_attr[:, :, None]
        sampled_attr = u*line_attr[:, 0] + (1 - u)*line_attr[:, 1]

    return np.hstack([sampled_point[:, :2], sampled_attr]) if line_attr is not None else sampled_point[:, :2]