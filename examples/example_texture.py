import numpy as np
from slam import texture
from slam import io as sio

if __name__ == '__main__':

    tex = sio.load_texture('data/example_texture.gii')
    print(tex)
    print(tex.metadata)
    print(tex.shape)
    print(tex.dtype)
    print(tex.min())
    print(tex.max())

    darray = np.zeros((2, 3))
    tex2 = texture.TextureND(darray=darray)
    print(tex2.metadata)
    print(tex2)
    print(tex2.shape)
    print(tex2.dtype)
    print(tex2.min())
    print(tex2.max())
    sio.write_texture(tex2, 'test.gii')
