'''
This file collects a bunch of helpers to make the transition between pinocchio2.7/hppfcl2.4
and pinocchio3x/hppfcl3x (version 2.99 and future).
- pin.GeometryModel as reversed init API 
- normals are opposite in hppfcl
'''

import pinocchio as pin
import hppfcl
import numpy as np

P3X = pin.__version__.split('.')[1] == '99'
HPPFCL3X = hppfcl.__version__.split('.')[1] == '99'

P3X = [int(n) for n in pin.__version__.split('.') ] >= [2,99]
HPPFCL3X = [int(n) for n in hppfcl.__version__.split('.') ] >= [2,99]

# -------------------------------------------------------------------------------
if not P3X:
    pin.ZAxis = np.array([0,0,1.])

# -------------------------------------------------------------------------------
# Monkey patch of computeDistances and computeCollisions to mimic p3x behavior

if not HPPFCL3X:
    pin._computeDistances = pin.computeDistances
    pin._computeCollisions = pin.computeCollisions

    def computeDistances(model,data,geometry_model,geometry_data,q):
        '''
        Mimic the behavior of computeDistances in pinocchio3x, by reversing the normals.
        '''
        pin._computeDistances(model,data,geometry_model,geometry_data,q)
        for d in geometry_data.distanceResults:
            d.normal *= -1

    def computeCollisions(model,data,geometry_model,geometry_data,q,stop_at_first_collision=False):
        '''
        Mimic the behavior of pin.computeCollisions by relying on computeDistances
        which is more generic in p2x.
        BIG LIMITATIONS: only one single contact point can be detected
        '''
        computeDistances(model,data,geometry_model,geometry_data,q)
        for p,c,d in zip(geometry_model.collisionPairs,geometry_data.collisionResults,geometry_data.distanceResults):
            c.clear()
            id1,id2 = p.first,p.second
            g1,g2 = geometry_model.geometryObjects[id1],geometry_model.geometryObjects[id2]
            contact = hppfcl.Contact(g1.geometry,g2.geometry,
                                     d.b1,d.b2,
                                     (d.getNearestPoint1()+d.getNearestPoint2())/2,
                                     d.normal,
                                     np.dot(d.normal,d.getNearestPoint2()-d.getNearestPoint1()))
            c.addContact(contact)

    computeDistances.__doc__ += '\n\nOriginal doc:\n' + pin._computeDistances.__doc__
    computeCollisions.__doc__ += '\n\nOriginal doc:\n' + pin._computeCollisions.__doc__
    pin.computeDistances = computeDistances
    pin.computeCollisions = computeCollisions

# -------------------------------------------------------------------------------
if not P3X:

    from enum import Enum
    class ContactType(Enum):
        CONTACT_3D = 3
        CONTACT_6D = 6

    class RigidConstraintData:
        pass

    class RigidConstraintModel:
        def __init__(self,
                     contactType,
                     model: pin.Model,
                     joint1_id,
                     joint1_placement,
                     joint2_id,
                     joint2_placement,
                     referenceFrame):
            assert(referenceFrame in pin.ReferenceFrame.values)
            assert(contactType in [ ContactType.CONTACT_3D, ContactType.CONTACT_6D ])

            self.contactType = contactType
            self.joint1_id = joint1_id
            self.joint1_placement = joint1_placement
            self.joint2_id = joint2_id
            self.joint2_placement = joint2_placement
            self.referenceFrame = referenceFrame

            self.name = ""

        def createData(self):
            return RigidConstraintData()

    pin.ContactType = ContactType
    pin.RigidConstraintModel = RigidConstraintModel
    pin.RigidConstraintData = RigidConstraintData
