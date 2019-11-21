import slam.io as sio

if __name__ == '__main__':

    mesh = sio.load_mesh('data/example_mesh.gii')
    # kdtree serves to compute distances to mesh vertices efficiently
    # here we compute the distance between a vector of two points and the mesh
    distance, index = mesh.kdtree.query([[0., 0., 0.], [0., 0., 1.]])
    print(distance, index)
