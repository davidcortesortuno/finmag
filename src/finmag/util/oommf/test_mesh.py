import numpy as np
from finmag.util.oommf import mesh

def test_iter_coords_int_zyx():
    m = mesh.Mesh((3, 1, 1), cellsize=(1, 1, 1))
    indices  = [r for r in m.iter_coords_int()]
    expected = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    assert np.array_equal(m.mesh_size, [3, 1, 1])
    assert m.array_order == mesh.Mesh.ZYX
    assert np.array_equal(expected, indices) 

    m = mesh.Mesh((2, 2, 2), cellsize=(1, 1, 1))
    indices  = [r for r in m.iter_coords_int()]
    expected = [[0,0,0], [1,0,0], [0,1,0], [1,1,0],
                [0,0,1], [1,0,1], [0,1,1], [1,1,1]]
    assert np.array_equal(m.mesh_size, [2, 2, 2])
    assert m.array_order == mesh.Mesh.ZYX
    assert np.array_equal(expected, indices) 
