import slam.io as sio
import slam.plot as splt
import slam.vertex_voronoi as svv
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')

if __name__ == '__main__':

    mesh = sio.load_mesh('example_mesh.gii')
    mesh.apply_transform(mesh.principal_inertia_transform)

    vert_vor = svv.vertexVoronoi(mesh)
    print(mesh.vertices.shape)
    print(vert_vor.shape)
    print(np.sum(vert_vor)-mesh.area)

    splt.pyglet_plot(mesh, vert_vor)
