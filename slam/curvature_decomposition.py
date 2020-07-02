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
