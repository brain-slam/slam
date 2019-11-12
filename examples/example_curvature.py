import trimesh
import numpy as np
import slam.curvature as get_curvatures

if __name__ == '__main__':
	
	# Generate a sphere
	mesh = trimesh.creation.icosphere()
	
	# Show th sphere
	mesh.show()
	
	# Calculate Rusinkiewicz estimation of mean and gauss curvatures
	PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = get_curvatures.getcurvaturesandderivatives(mesh)
	gaussian_curv = PrincipalCurvatures[0, :] * PrincipalCurvatures[1, :]
	mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])
	
	# Plot mean curvature
	vect_col_map = \
		trimesh.visual.color.interpolate(mean_curv, color_map='jet')
	
	if mean_curv.shape[0] == mesh.vertices.shape[0]:
		mesh.visual.vertex_colors = vect_col_map
	elif mean_curv.shape[0] == mesh.faces.shape[0]:
		mesh.visual.face_colors = vect_col_map
	mesh.show( background=[0, 0, 0, 255])
	
	# PLot Gauss curvature
	vect_col_map = \
		trimesh.visual.color.interpolate(gaussian_curv, color_map='jet')
	if gaussian_curv.shape[0] == mesh.vertices.shape[0]:
		mesh.visual.vertex_colors = vect_col_map
	elif gaussian_curv.shape[0] == mesh.faces.shape[0]:
		mesh.visual.face_colors = vect_col_map
	mesh.show(background=[0, 0, 0, 255])

