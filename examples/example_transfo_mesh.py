import slam.io as sio
import slam.differential_geometry as sdg
import numpy as np
from trimesh import transformations
mesh_file = "../examples/data/example_mesh.gii"

mesh = sio.load_mesh(mesh_file)

vertices = mesh.vertices

theta = np.pi / 2
rot_x = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta),  np.cos(theta)]])



#############################################################################
# VISUALIZATION USING INTERNAL TOOLS
#############################################################################
import slam.plot as splt

# Plot Original Mesh
display_settings = {}
mesh_data = {}
mesh_data['vertices'] = vertices
mesh_data['faces'] = mesh.faces
mesh_data['title'] = 'example_mesh.gii Without Transformation'

intensity_data = None
Fig = splt.mes3d_projection(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
Fig.show()

mesh.apply_transform(mesh.principal_inertia_transform)

display_settings = {}
mesh_data = {}
mesh_data['vertices'] = mesh.vertices
mesh_data['faces'] = mesh.faces
mesh_data['title'] = 'example_mesh.gii PCA Transformation'
intensity_data = None
Fig = splt.mes3d_projection(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
Fig.show()

vertices_translate = np.dot(rot_x, mesh.vertices.T).T

display_settings = {}
mesh_data = {}
mesh_data['vertices'] = vertices_translate
mesh_data['faces'] = mesh.faces
mesh_data['title'] = 'example_mesh.gii With PCA and rotation Transformation'
intensity_data = None
Fig = splt.mes3d_projection(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
Fig.show()