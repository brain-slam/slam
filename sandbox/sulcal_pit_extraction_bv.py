# -*- coding: utf-8 -*-
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license version 2 and that you accept its terms.


def validation():
    try:
        from brainvisa.cortical_surface.surface_tools import PDE_tools as pdeTls
        from brainvisa.cortical_surface.surface_tools import mesh_watershed as watershed
    except:
        raise ValidationError('brainvisa.cortical_surface.surface_tools module can not be imported.')


from brainvisa.processes import *
import numpy as np

try:
    from brainvisa.cortical_surface.surface_tools import PDE_tools as pdeTls
    from brainvisa.cortical_surface.surface_tools import mesh_watershed as watershed
except:
    pass

name = 'Sulcal Pits Extraction'
userLevel = 0

# Argument declaration
signature = Signature(
    'input_mesh', ReadDiskItem('Hemisphere White Mesh', 'Aims mesh formats'),
    'side', Choice('left', 'right'),
    'mask_texture', ReadDiskItem('Cingular pole texture', 'Aims Texture formats'),
    'DPF_alpha', Float(),
    'thresh_ridge', Float(),
    'thresh_dist', Float(),
    'group_average_Fiedler_length', Float(),
    'thresh_area', Float(),
    'group_average_surface_area', Float(),
    'DPF_texture', WriteDiskItem('DPF texture', 'Aims texture formats'),
    'pits_texture', WriteDiskItem('pits texture', 'Aims texture formats'),
    'noisypits_texture', WriteDiskItem('noisy pits texture', 'Aims texture formats'),
    'ridges_texture', WriteDiskItem('ridges texture', 'Aims texture formats'),
    'basins_texture', WriteDiskItem('basins texture', 'Aims texture formats'),
    #    'areas_texture',WriteDiskItem( ' texture',  'Aims texture formats' ),
)


# Default values
def initialization(self):
    def linkSide(proc, dummy):
        if proc.input_mesh is not None:
            return proc.input_mesh.get('side')

    self.linkParameters('side', 'input_mesh', linkSide)
    self.linkParameters('DPF_texture', 'input_mesh')
    self.linkParameters('mask_texture', 'input_mesh')
    self.linkParameters('pits_texture', 'input_mesh')
    self.linkParameters('noisypits_texture', 'input_mesh')
    self.linkParameters('ridges_texture', 'input_mesh')
    self.linkParameters('basins_texture', 'input_mesh')
    self.DPF_alpha = 0.03
    self.thresh_dist = 20
    self.thresh_ridge = 1.5
    self.thresh_area = 50

    # default values were computed across 137 subjects from the OASIS database, for details, see:
    #  Auzias, G., Brun, L., Deruelle, C., & Coulon, O. (2015). Deep sulcal landmarks: Algorithmic and conceptual improvements in
    #  the definition and extraction of sulcal pits. NeuroImage, 111, 12â25. doi:10.1016/j.neuroimage.2015.02.008
    # values from this paper:
    # all 137 subjects :  area_L 91369, fiedler_L 236, area_R 91434, fiedler_R 238
    # group1 area_L 91665, fiedler_L 237, area_R 91742, fiedler_R 238
    # group2 area_L 91078, fiedler_L 235, area_R 91130, fiedler_R 239

    # the default values given in the current process were recomputed from the 137 subjects
    # slightly different values are due to implementations details.

    def linkSurfaceSide(proc, dummy):
        if proc.input_mesh is not None:
            side = proc.input_mesh.get('side')
            if side == 'left':
                return 91369.33
            else:
                return 91433.68

    self.linkParameters('group_average_surface_area', 'input_mesh', linkSurfaceSide)

    def linkFiedlerSide(proc, dummy):
        if proc.input_mesh is not None:
            side = proc.input_mesh.get('side')
            if side == 'left':
                return 235.95
            else:
                return 238.25

    self.linkParameters('group_average_Fiedler_length', 'input_mesh', linkFiedlerSide)
    self.setOptional('mask_texture')


def execution(self, context):
    re = aims.Reader()
    ws = aims.Writer()

    mesh = re.read(self.input_mesh.fullPath())

    # compute the vertex_voronoi of the mesh that will be used in the  watershed hereafter
    vert_voronoi = pdeTls.vertexVoronoi(mesh)

    # compute the DPF
    tmp_curv_tex = context.temporary('Texture')
    context.write('computing DPF')
    context.system('AimsMeshCurvature', '-i', self.input_mesh, '-o', tmp_curv_tex.fullPath(), '-m', 'fem')
    curv = re.read(tmp_curv_tex.fullPath())
    k = curv[0].arraydata()
    dpf = pdeTls.depthPotentialFunction(mesh, k, [self.DPF_alpha])
    tex_dpf = aims.TimeTexture_FLOAT()
    tex_dpf[0].assign(dpf[0])
    ws.write(tex_dpf, self.DPF_texture.fullPath())
    context.write('DPF done')
    # apply the watershed
    context.write('Computing the Fiedler geodesic length and surface area')

    mesh_area = np.sum(vert_voronoi)
    (mesh_fiedler_length, fiedler_tex) = pdeTls.meshFiedlerLength(mesh, 'geodesic')

    if self.mask_texture is not None:
        maskTex = re.read(self.mask_texture.fullPath())
        mask = np.array(maskTex[0])
    else:
        mask = np.zeros(dpf[0].shape)


    # Normalization of watershed merging parameters
    thresh_dist = self.thresh_dist * mesh_fiedler_length / self.group_average_Fiedler_length
    thresh_area = self.thresh_area * mesh_area / self.group_average_surface_area

    # Watershed
    # first step : merging online
    context.write('Computing the watershed with distance and ridge criteria for basins merging')
    labels_1, pitsKept_1, pitsRemoved_1, ridgePoints, parent_1 = watershed.watershed(mesh, vert_voronoi, dpf[0], mask,
                                                                                     thresh_dist, self.thresh_ridge)

    # second step : merging offline
    context.write('basins merging based on area criterion')
    labels, infoBasins, pitsKept, pitsRemoved_2, parent = watershed.areaFiltering(mesh, vert_voronoi, labels_1,
                                                                                  pitsKept_1, parent_1, thresh_area)
    pitsRemoved = pitsRemoved_1 + pitsRemoved_2

    # Saving
    # texture of basins
    labelsTexture = aims.TimeTexture_S16(1, len(labels))
    labelsTexture[0].assign(labels)
    ws.write(labelsTexture, self.basins_texture.fullPath())
    # texture of pits
    atex_pits = np.zeros((len(labels), 1))
    for pit in pitsKept:
        atex_pits[pit[0]] = 1
    pitsTexture = aims.TimeTexture_S16(1, len(labels))
    pitsTexture[0].assign(atex_pits)
    ws.write(pitsTexture, self.pits_texture.fullPath())
    # texture of noisy pits
    atex_noisypits = np.zeros((len(labels), 1))
    for pit in pitsRemoved:
        atex_noisypits[pit[0]] = 1
    noisypitsTexture = aims.TimeTexture_S16(1, len(labels))
    noisypitsTexture[0].assign(atex_noisypits)
    ws.write(noisypitsTexture, self.noisypits_texture.fullPath())
    # texture of ridges
    atex_ridges = np.zeros((len(labels), 1))
    for ridge in ridgePoints:
        atex_ridges[ridge[2]] = 1
    ridgesTexture = aims.TimeTexture_S16(1, len(labels))
    ridgesTexture[0].assign(atex_ridges)
    ws.write(ridgesTexture, self.ridges_texture.fullPath())