import numpy as np
import slam.curvature as scurv
import slam.generate_parametric_surfaces as sgps

# Implementation based on
# " Surface shape and curvature scales
# Jan JKoenderink & Andrea Jvan Doorn 
# Image and Vision Computing
# Volume 10, Issue 8, October 1992, Pages 557-564 "

def decompose_curvature(curvatures):
    curvatures = np.sort(curvatures, 1)[::-1]
    shapeIndex = (2 / np.pi) * np.arctan(
        (curvatures[0, :] + curvatures[1, :]) /
        (curvatures[1, :] - curvatures[0, :])
    )
    curvedness = np.sqrt((curvatures[0, :]**2 + curvatures[1, :]**2) / 2)
    return shapeIndex, curvedness

def curvedness_shapeIndex(mesh):
    curv = scurv.curvatures_and_derivatives(mesh)[0]
    return decompose_curvature(curv)


import slam.io as sio
import slam.plot as splt
import slam.curvature as scurv

mesh_a = sio.load_mesh("../examples/data/example_mesh.gii")


K = [1, 1]
quadric = sgps.generate_quadric(K, nstep=20, ax=3, ay=1,
                                random_sampling=True,
                                ratio=0.3,
                                random_distribution_type='gamma')
quadric_mean_curv = \
    sgps.quadric_curv_mean(K)(np.array(quadric.vertices[:, 0]),
                              np.array(quadric.vertices[:, 1]))

#mesh_a = quadric

pcurv = scurv.curvatures_and_derivatives(mesh_a)[0]
mean_curv = .5 * (pcurv[0, :] + pcurv[1, :])
shapeIndex, curvedness = decompose_curvature(pcurv)

visb_sc = splt.visbrain_plot(mesh=mesh_a, tex=mean_curv,
                             caption='mean curvature',)

visb_sc = splt.visbrain_plot(mesh=mesh_a, tex=shapeIndex,
                             caption='shapeIndex',visb_sc=visb_sc)

visb_sc = splt.visbrain_plot(mesh=mesh_a, tex=curvedness,
                             caption='curvedness', visb_sc=visb_sc)
visb_sc.preview()