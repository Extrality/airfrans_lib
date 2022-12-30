import numpy as np

def reorganize(in_order_points, out_order_points, quantity_to_reorder):
    """
    Reorder points coming from different source of data with the same positions.
    
    Args:
        in_order_points (np.ndarray): Positions of the features we want to reorder (shape `(N, 2)`).
        out_order_points (np.ndarray): Positions for the new ordering (shape `(N, 2)`).
        quantity_to_reorder (np.ndarray): Features attached to the points we want to reorder (shape `(N, F)`).
    """
    n = out_order_points.shape[0]
    idx = np.zeros(n)
    for i in range(n):
        cond = (out_order_points[i] == in_order_points)
        cond = cond[:, 0]*cond[:, 1]        
        idx[i] = np.argwhere(cond)[0][0]
    idx = idx.astype('int')

    assert (in_order_points[idx] == out_order_points).all()

    return quantity_to_reorder[idx]