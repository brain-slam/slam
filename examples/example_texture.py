import numpy as np
from slam import texture
from slam import io as sio

if __name__ == '__main__':

    tex = sio.load_texture('example_texture.gii')
    print(tex)
    print(tex.shape)
    print(tex.dtype)

    darray = np.zeros((2, 3))
    tex2 = texture.TextureND(darray=darray)
    print(tex2)
    print(tex2.shape)
    print(tex2.dtype)
