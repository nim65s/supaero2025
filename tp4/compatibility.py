'''
This file collects a bunch of helpers to make the transition between pinocchio2.7/hppfcl2.4
and pinocchio3x/hppfcl3x (version 2.99 and future).
- pin.GeometryModel as reversed init API 
- normals are opposite in hppfcl
'''

import pinocchio as pin
import hppfcl

P3X = pin.__version__.split('.')[1] == '99'
HPPFCL3X = hppfcl.__version__.split('.')[1] == '99'

if HPPFCL3X:
    def hppfcl_normals(n):
        '''
        This functions is introduced for compatibility with earlier versions of 
        HPP-FCL. With versions 3x, this function is the identity (return the input).
        '''
        return n
else:
    def hppfcl_normals(n):
        '''
        This functions is introduced for compatibility with earlier versions of 
        HPP-FCL. 
        The version returns the opposite of the input normal vector.
        In version HPP-FCL 2x, the normal was defined in the wrong direction (i.e from
        point P2 to P1). In the 3x version, it is the opposite. We align with the
        P3X convention.
        '''
        return -n
