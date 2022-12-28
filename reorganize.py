import torch

def reorganize(in_order_points, out_order_points, quantity_to_reorder):
    """
    Reorder points coming from different source of data with the same positions.
    
    Args:
        in_order_points (torch.tensor): Positions of the features we want to reorder (shape (N, 2)).
        out_order_points (torch.tensor): Positions for the new ordering (shape (N, 2)).
        quantity_to_reorder (torch.tensor): Features attached to the points we want to reorder (shape (N, F)).
    """
    n = out_order_points.shape[0]
    idx = torch.zeros(n)
    for i in range(n):
        cond = (out_order_points[i] == in_order_points)
        cond = cond[:, 0]*cond[:, 1]        
        idx[i] = torch.argwhere(cond)[0][0]
    idx = idx.long()

    assert (in_order_points[idx] == out_order_points).all()

    return quantity_to_reorder[idx]