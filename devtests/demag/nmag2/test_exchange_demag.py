import os
import cProfile
import pstats
import finmag.sim.helpers as h

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
TOLERANCE = 1e-3

def test_compare_averages():
    ref = h.read_float_data(MODULE_DIR + "/averages_ref.txt")
    computed = h.read_float_data(MODULE_DIR + "/averages.txt")

    for i in range(len(computed)):
        t_ref, mx_ref, my_ref, mz_ref = ref[i]
        t, mx, my, mz = computed[i]

        assert abs(mx - mx_ref) < TOLERANCE
        assert abs(my - my_ref) < TOLERANCE
        assert abs(mz - mz_ref) < TOLERANCE
