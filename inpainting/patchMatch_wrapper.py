import ctypes
import os

basedir = os.path.abspath(os.path.dirname(__file__))
print(basedir)

def inpainting(imgPath, maskPath, outputPath):
    img = imgPath.encode('utf-8')
    mask = maskPath.encode('utf-8')
    output = outputPath.encode('utf-8')
    patchMatch_lib = ctypes.CDLL(basedir + "build/libpatchMatch.so")
    patchMatch = patchMatch_lib.patchMatch
    patchMatch.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
    patchMatch(img, mask, output)