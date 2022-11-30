"""
.. _example_differential_geometry:

===================================
example of differential geometry tools in slam
===================================
"""

# Authors: Guillaume Auzias <guillaume.auzias@univ-amu.fr>

# License: BSD (3-clause)
# sphinx_gallery_thumbnail_number = 2


###############################################################################
# importation of slam modules
import slam.io as sio
import slam.differential_geometry as sdg
import slam.plot as splt

###############################################################################
# loading an examplar mesh and corresponding texture and show it
mesh_file = "../examples/data/example_mesh.gii"
texture_file = "../examples/data/example_texture.gii"
mesh = sio.load_mesh(mesh_file)
tex = sio.load_texture(texture_file)

visb_sc = splt.visbrain_plot(
    mesh=mesh, tex=tex.darray[0], caption="mesh with curvature", cblabel="curvature"
)
visb_sc.preview()

###############################################################################
# compute various types of Laplacian of the mesh
lap, lap_b = sdg.compute_mesh_laplacian(mesh, lap_type="fem")
print(mesh.vertices.shape)
print(lap.shape)
lap, lap_b = sdg.compute_mesh_laplacian(mesh, lap_type="conformal")
lap, lap_b = sdg.compute_mesh_laplacian(mesh, lap_type="meanvalue")
lap, lap_b = sdg.compute_mesh_laplacian(mesh, lap_type="authalic")

###############################################################################
# smooth the mesh using Laplacian
s_mesh = sdg.laplacian_mesh_smoothing(mesh, nb_iter=100, dt=0.1)

###############################################################################
# show it
visb_sc = splt.visbrain_plot(mesh=s_mesh, caption="smoothed mesh")
visb_sc.preview()

###############################################################################
# compute the gradient of texture tex
triangle_grad = sdg.triangle_gradient(mesh, tex.darray[0])
print(triangle_grad)
grad = sdg.gradient(mesh, tex.darray[0])
print(grad)
norm_grad = sdg.norm_gradient(mesh, tex.darray[0])
print(norm_grad)

###############################################################################
# show it
visb_sc = splt.visbrain_plot(
    mesh=mesh,
    tex=norm_grad,
    caption="norm of the gradient of curvature",
    cblabel="gradient magnitude",
)
visb_sc.preview()

###############################################################################
# compute the depth potential function
dpf = sdg.depth_potential_function(mesh, tex.darray[0], [0.3])

###############################################################################
# show it
visb_sc = splt.visbrain_plot(
    mesh=mesh, tex=dpf[0], caption="depth potential function", cblabel="dpf"
)
visb_sc.preview()
