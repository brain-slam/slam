import slam.io as sio

if __name__ == '__main__':

    mesh = sio.load_mesh('data/example_mesh.gii')
    distance, index = mesh.kdtree.query([[0., 0., 0.], [0., 0., 1.]])
    print(distance, index)
