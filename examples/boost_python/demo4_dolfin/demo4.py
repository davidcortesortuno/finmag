import dolfin as df
import demo4_module

mesh = df.UnitCube(5, 5, 5)
print "Number of vertices:", demo4_module.get_num_vertices(mesh)
