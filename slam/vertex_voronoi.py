import numpy as np
import itertools
import slam.io as sio
import slam.texture as stex
import os
import time
import pandas as pd

def voronoi_de_papa(mesh):
    """
    @author : Maxime Dieudonné maxime.dieudonne@univ-amu.fr
    compute vertex voronoi of a mesh as described in
    Meyer, M., Desbrun, M., Schröder, P., & Barr, A. (2002).
    Discrete differential-geometry operators for triangulated 2-manifolds.
    Visualization and Mathematics, 1–26.
    Voronoi texture : Area of Voronoi at each vertex.
    @param mesh: a Trimesh mesh object
    @return: a ndarray of the voronoi texture.
    """
    Nbp = mesh.faces.shape[0]
    obt_angs = mesh.face_angles > np.pi / 2
    obt_poly = obt_angs[:, 0] | obt_angs[:, 1] | obt_angs[:, 2]
    obt_poly_df = np.array([[x]*3 for x in obt_poly]).flatten()
    area_face = mesh.area_faces
    area_face_df = np.array([[x]*3 for x in area_face]).flatten()
    print('    -percent polygon with obtuse angle ', 100.0 * len(np.where(obt_poly)[0]) / Nbp)
    cot = 1 / np.tan(mesh.face_angles)
    #
    comb = np.array([list(itertools.combinations(mesh.faces[i], 2)) for i in range(Nbp)])
    comb = comb.reshape([comb.size // 2, 2])
    V1 = np.array([mesh.vertices[i] for i in comb[:, 0]])
    V2 = np.array([mesh.vertices[i] for i in comb[:, 1]])
    #
    # Dist array : compute the sqare of the norm of each segments.We need to switch the first and the last colums to gets athe end
    # the right order : D0, D1, D2
    # faces | segments of the faces
    # F0    | D0 D1 D2
    # F1    | D0 D1 D2
    # FN    | D0 D1 D2
    Dist = np.sum(np.power(V1 - V2, 2), axis=1)
    Dist = Dist.reshape([Dist.size // 3, 3])
    Dist[:, [2, 0]] = Dist[:, [0, 2]]
    #
    # VorA0 : compute the voronoi area (hypothesis : the face triangle is Aigu) for all the V0 of the faces
    # VorA1 : compute the voronoi area (hypothesis : the face triangle is Aigu) for all the V1 of the faces
    # VorA2 : compute the voronoi area (hypothesis : the face triangle is Aigu) for all the V2 of the faces
    # VorA : concatenation of VorA0, VorA1, VorA2.
    # faces | segments of the faces
    # F0    | VorA0 VorA1 VorA2
    # F1    | VorA0 VorA1 VorA2
    # FN    | VorA0 VorA1 VorA2
    VorA0 = Dist[:, 1] * cot[:, 1] + Dist[:, 2] * cot[:, 2]
    VorA1 = Dist[:, 0] * cot[:, 0] + Dist[:, 2] * cot[:, 2]
    VorA2 = Dist[:, 0] * cot[:, 0] + Dist[:, 1] * cot[:, 1]
    VorA = np.array([VorA0, VorA1, VorA2]).transpose()
    # df_Vor
    # faces : the index of the faces (it doesn't appear in the dataframe)
    # Vertex : the index of the vertex
    # VorA : the Voronoi area with the hypothesis "the triangle face is aigu"
    # Area : the area of the face (usefull for the next because the voronoi area in case the triangle is obtu depends of it)
    # Fobt : Face obtu true or false
    # Aobt : Angle Obtu true or false
    #
    # faces | Vertex    | VorA  | Area  |Fobt   |Aobt
    # F0    | V0        |   0.2 |  1.2  | true  | false
    # F0    | V1        |
    # F0    | V2        |
    # ...
    # FN    | V0        |
    # FN    | V1        |
    # FN    | V2        |
    VorA = VorA.flatten()
    face_flat = mesh.faces.flatten()
    df_Vor = pd.DataFrame(dict(VorA=VorA,Area = area_face_df, Vertex=face_flat, Fobt = obt_poly_df, Aobt = obt_angs.flatten()))
    # we create 3 subdataframe according the condition on Fobt and Aobt. The 3 possibilities are :
    #  (Fobt, Aobt) = (False, False) -> the voronoi area of a vertex in 1 triangle face is given in VorA
    #  (Fobt, Aobt) = (True, False)  -> the voronoi area of a vertex in 1 triangle face is given by area_face/4
    #  (Fobt, Aobt) = (True, True)  -> the voronoi area of a vertex in 1 triangle face is given by area_face/2
    # area_VorA : sum of the vornoi area only for vertex with angle in  aigue faces
    # area_VorOA : sum of the vornoi area only for vertex with Aigue angle in  Obtue faces
    # area_VorOO : sum of the vornoi area only for vertex with Obtue angle in  Obtue faces
    # area_VorA
    area_VorA = df_Vor[df_Vor['Fobt']==False].groupby('Vertex',as_index=False).sum()[['Vertex','VorA']]
    area_VorA['VorA'] = area_VorA['VorA']/8
    # area_VorOA
    area_VorOA = df_Vor[(df_Vor['Fobt'] == True) & (df_Vor['Aobt'] == False)].groupby('Vertex',as_index=False).sum()[['Vertex','Area']]
    area_VorOA.rename(columns={'Area':'A4'}, inplace=True)
    area_VorOA['A4'] = area_VorOA['A4']/4
    # area_VorOO
    Area_VorOO = df_Vor[(df_Vor['Fobt'] ==True) & (df_Vor['Aobt'] == True)].groupby('Vertex',as_index=False).sum()[['Vertex','Area']]
    Area_VorOO.rename(columns={'Area':'A2'}, inplace=True)
    Area_VorOO['A2'] = Area_VorOO['A2']/2
    # Then the voronoi area of one vertex in the mesh is given by the sum of all the vornoi area in its neighboored triangle faces
    # so we sum for each vertex the area given in area_VorA,area_VorOA and area_VorOO if the value exist
    #area_Vor2 :
    # | idx Vertex      | VorA   | A2    | A4  |
    # | 1               |   0.2 |  1.2  | Nan |  -> sum = area_vor
    # | 2               |   0.7 |  Nan  | Nan |  -> sum
    # | ...             |
    # | 480 562         |                        -> sum
    df_area_Vor = pd.merge(area_VorA, area_VorOA, on='Vertex', how="outer")
    df_area_Vor2 = pd.merge(df_area_Vor, Area_VorOO, on='Vertex', how="outer")
    df_area_Vor2 = df_area_Vor2.sort_values(['Vertex'])
    area_Vor = df_area_Vor2[['VorA', 'A2', 'A4']].sum(axis=1)
    return area_Vor.values
