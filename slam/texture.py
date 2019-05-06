"""
github.com/gauzias/slam
----------------------------

definition of the Texture class
"""

import numpy as np


class TextureND:

    def __init__(self,
                 darray=None,
                 process=True,
                 metadata=None,
                 **kwargs):
        """
        A Trimesh object contains a triangular 3D mesh.

        Parameters
        ----------
        metadata : dict
          Any metadata about the mesh
        process : bool
          if True, Nan and Inf values will be removed

        """
        if darray is None:
            self.darray = None
            self.dtype = None
            self.shape = None
            self.is_empty = True
        else:
            # (n, 3) float, set of vertices
            self.darray = np.asanyarray(darray)
            self.dtype = self.darray.dtype
            self.shape = self.darray.shape

        # store metadata about the texture in a dictionary
        self.metadata = dict()
        # update the mesh metadata with passed metadata

        if isinstance(metadata, dict):
            self.metadata.update(metadata)

        if self.shape is not None:
            self.is_empty = False

        # process will remove NaN and Inf values and merge vertices
        # if validate, will remove degenerate and duplicate faces
        if process:
            self.process()

        # save reference to kwargs
        self._kwargs = kwargs

    def process(self):
        """
        Do the bare minimum processing to make a mesh useful.

        Does this by:
            1) removing NaN and Inf values

            2) merging duplicate vertices

        If self._validate:
            3) Remove triangles which have one edge of their rectangular 2D
               oriented bounding box shorter than tol.merge

            4) remove duplicated triangles

        Returns
        ------------
        self: trimesh.Trimesh
          Current mesh
        """
        # if there are no vertices or faces exit early
        if self.is_empty:
            return self

        self.remove_infinite_values()
        self.metadata['processed'] = True

    def remove_infinite_values(self):
        """
        Ensure that every data_array consists of finite numbers.

        This will remove np.nan and np.inf
        """

        # if the texture is already empty we can't remove anything
        if self.is_empty:
            return

        ind_infinite = np.isfinite(self.darray)
        darray = self.darray.copy()
        darray[ind_infinite] = 0
        self.update_darray(darray)

    def update_darray(self, darray):
        """
        Update darray and dtype, shape accordingly.
        :param darray:
        :return: Current TextureND
        """

        self.darray = np.asanyarray(darray)
        self.dtype = self.darray.dtype
        self.shape = self.darray.shape
        if self.shape is not None:
            self.is_empty = False

    def copy(self):
        """
        Safely get a copy of the current texture.

        Returns
        ---------
        copied : TextureND
          Copy of current texture
        """
        copied = TextureND()
        copied.update_darray(self.darray)
        copied.metadata = self.metadata.copy()

        return copied

    def min(self):
        return self.darray.min()

    def max(self):
        return self.darray.max()

    def mean(self):
        return self.darray.mean()
