"""
This file collects a bunch of helpers to make the transition between
pinocchio2.7/hppfcl2.4 and pinocchio3x/hppfcl3x (version 2.99 and future).
- pin.GeometryModel as reversed init API
- normals are opposite in hppfcl

Should be deprecated now, can be removed.
"""

import warnings

import hppfcl
import numpy as np
import pinocchio as pin
from numpy.linalg import norm

P3X = pin.__version__.split(".")[1] == "99"
HPPFCL3X = hppfcl.__version__.split(".")[1] == "99"

P3X = [int(n) for n in pin.__version__.split(".")] >= [2, 99]
HPPFCL3X = [int(n) for n in hppfcl.__version__.split(".")] >= [2, 99]

assert P3X and HPPFCL3X

# -------------------------------------------------------------------------------
if not P3X:
    pin.ZAxis = np.array([0, 0, 1.0])

# -------------------------------------------------------------------------------
# Monkey patch of computeDistances and computeCollisions to mimic p3x behavior

if not HPPFCL3X:
    pin._computeDistances = pin.computeDistances
    pin._computeCollisions = pin.computeCollisions

    def refineDistance_SphSph(g1, g2, oMg1, oMg2, res):
        o1o2 = oMg2.translation - oMg1.translation
        res.normal = o1o2 / norm(o1o2)

    def refineDistance_SphPlane(g1, g2, oMg1, oMg2, res):
        p1p2 = res.getNearestPoint2() - res.getNearestPoint1()
        normal = oMg2.rotation @ g2.geometry.n
        assert np.allclose(np.cross(p1p2, normal), 0)
        res.normal = -normal
        # res.normal = -normal if res.min_distance<0 else normal
        # res.normal = normal if np.dot(p1p2,normal)>=0 else -normal

    def refineDistance_BoxPlane(g1, g2, oMg1, oMg2, res):
        # This one is not working
        p1p2 = res.getNearestPoint2() - res.getNearestPoint1()
        normal = oMg2.rotation @ g2.geometry.n
        assert np.allclose(np.cross(p1p2, normal), 0)
        # res.normal = -normal if d.min_distance<0 else normal
        res.normal = -normal

    def computeDistances(model, data, geometry_model, geometry_data, q):
        """
        Mimic the behavior of computeDistances in pinocchio3x, by reversing the normals.
        """
        pin._computeDistances(model, data, geometry_model, geometry_data, q)
        for ip, d in enumerate(geometry_data.distanceResults):
            pair = geometry_model.collisionPairs[ip]
            i1, i2 = pair.first, pair.second
            g1, g2 = (
                geometry_model.geometryObjects[i1],
                geometry_model.geometryObjects[i2],
            )
            sh1, sh2 = g1.geometry, g2.geometry
            oMg1, oMg2 = geometry_data.oMg[i1], geometry_data.oMg[i2]

            # Sphere Sphere
            if isinstance(sh1, hppfcl.Sphere) and isinstance(sh2, hppfcl.Sphere):
                refineDistance_SphSph(g1, g2, oMg1, oMg2, d)

            # Box box
            elif isinstance(sh1, hppfcl.Box) and isinstance(sh2, hppfcl.Box):
                # refineDistance_BoxBox(g1,g2,oMg1,oMg2,d)
                # if d.min_distance<1e-3: stop
                print("Box-box collisions not working in P2X")
                assert False and "Box-box collisions not working in P2X"

            # Sphere Plane
            elif isinstance(sh1, hppfcl.Sphere) and isinstance(sh2, hppfcl.Halfspace):
                refineDistance_SphPlane(g1, g2, oMg1, oMg2, d)
            elif isinstance(sh1, hppfcl.Halfspace) and isinstance(sh2, hppfcl.Sphere):
                refineDistance_SphPlane(g2, g1, oMg2, oMg1, d)

            # Box Plane
            elif isinstance(sh1, hppfcl.Box) and isinstance(sh2, hppfcl.Halfspace):
                refineDistance_BoxPlane(g1, g2, oMg1, oMg2, d)
            elif isinstance(sh1, hppfcl.Halfspace) and isinstance(sh2, hppfcl.Box):
                refineDistance_BoxPlane(g2, g1, oMg2, oMg1, d)

            # Misc
            else:
                if not np.any(np.isnan(d.normal)):
                    d.normal *= -1
                    # Check normal against witness direction, just to be sure
                    witness = d.getNearestPoint2() - d.getNearestPoint1()
                    w = np.linalg.norm(witness)
                    if w > 1e-5:
                        if not np.allclose(witness / w, d.normal):
                            msg = (
                                f"Normal not aligned with witness segment (pair {ip} "
                                + f"{type(sh1)}-{type(sh2)})"
                            )
                            warnings.warn(msg, category=UserWarning, stacklevel=2)
                else:
                    # Poor patch, not working in penetration
                    print("# Poor patch, not working in penetration", ip, sh1, sh2)
                    msg = (
                        f"Setting normals from witness segment (pair {ip} "
                        + f"{type(sh1)}-{type(sh2)}) ### Poor patch, "
                        + "not working in penetration"
                    )
                    warnings.warn(msg, category=UserWarning, stacklevel=2)
                    witness = d.getNearestPoint2() - d.getNearestPoint1()
                    w = np.linalg.norm(witness)
                    assert w > 1e-5
                    d.normal = witness / w

    def computeCollisions(
        model, data, geometry_model, geometry_data, q, stop_at_first_collision=False
    ):
        """
        Mimic the behavior of pin.computeCollisions by relying on computeDistances
        which is more generic in p2x.
        BIG LIMITATIONS: only one single contact point can be detected
        """
        isInCollision = False
        computeDistances(model, data, geometry_model, geometry_data, q)
        for p, cr, c, d in zip(
            geometry_model.collisionPairs,
            geometry_data.collisionRequests,
            geometry_data.collisionResults,
            geometry_data.distanceResults,
        ):
            c.clear()
            id1, id2 = p.first, p.second
            g1, g2 = (
                geometry_model.geometryObjects[id1],
                geometry_model.geometryObjects[id2],
            )
            # dist = np.dot(d.normal,d.getNearestPoint2()-d.getNearestPoint1())
            dist = d.min_distance
            if dist < cr.security_margin:
                contact = hppfcl.Contact(
                    g1.geometry,
                    g2.geometry,
                    d.b1,
                    d.b2,
                    (d.getNearestPoint1() + d.getNearestPoint2()) / 2,
                    d.normal,
                    dist,
                )
                c.addContact(contact)
                isInCollision = True
        return isInCollision

    computeDistances.__doc__ += "\n\nOriginal doc:\n" + pin._computeDistances.__doc__
    computeCollisions.__doc__ += "\n\nOriginal doc:\n" + pin._computeCollisions.__doc__
    pin.computeDistances = computeDistances
    pin.computeCollisions = computeCollisions

# -------------------------------------------------------------------------------
if not HPPFCL3X:
    hppfcl.hppfcl.Contact.getNearestPoint1 = lambda self: self.pos
    hppfcl.hppfcl.Contact.getNearestPoint2 = lambda self: self.pos

# -------------------------------------------------------------------------------
if not P3X:
    from enum import Enum

    class ContactType(Enum):
        CONTACT_3D = 3
        CONTACT_6D = 6

    class RigidConstraintData:
        pass

    class RigidConstraintModel:
        def __init__(
            self,
            contactType,
            model: pin.Model,
            joint1_id,
            joint1_placement,
            joint2_id,
            joint2_placement,
            referenceFrame,
        ):
            assert referenceFrame in pin.ReferenceFrame.values
            assert contactType in [ContactType.CONTACT_3D, ContactType.CONTACT_6D]

            self.type = contactType
            self.joint1_id = joint1_id
            self.joint1_placement = joint1_placement
            self.joint2_id = joint2_id
            self.joint2_placement = joint2_placement
            self.reference_frame = referenceFrame

            self.name = ""

        def createData(self):
            return RigidConstraintData()

    pin.ContactType = ContactType
    pin.RigidConstraintModel = RigidConstraintModel
    pin.RigidConstraintData = RigidConstraintData

# -------------------------------------------------------------------------------
# Monkey patch of getConstraintsJacobian


def _getConstraintJacobian3d(model, data, contact_model, contact_data):
    """
    Returns the constraint Jacobian for 3d contact
    """
    assert contact_model.type == pin.ContactType.CONTACT_3D
    assert contact_model.reference_frame == pin.LOCAL
    c1_J_j1 = contact_model.joint1_placement.inverse().action[
        :3
    ] @ pin.getJointJacobian(model, data, contact_model.joint1_id, pin.LOCAL)
    c2_J_j2 = contact_model.joint2_placement.inverse().action[
        :3
    ] @ pin.getJointJacobian(model, data, contact_model.joint2_id, pin.LOCAL)
    c1Rc2 = (
        contact_model.joint1_placement.rotation.T
        @ data.oMi[contact_model.joint1_id].rotation.T
        @ data.oMi[contact_model.joint2_id].rotation
        @ contact_model.joint2_placement.rotation
    )

    J = c1_J_j1
    J -= c1Rc2 @ c2_J_j2

    return J


def _getConstraintJacobian6d(model, data, contact_model, contact_data):
    """
    Returns the constraint Jacobian
    """
    assert contact_model.type == pin.ContactType.CONTACT_6D
    assert contact_model.reference_frame == pin.LOCAL
    c1_J_j1 = contact_model.joint1_placement.inverse().action @ pin.getJointJacobian(
        model, data, contact_model.joint1_id, pin.LOCAL
    )
    c1_J_j2 = (
        contact_model.joint1_placement.inverse().action
        @ data.oMi[contact_model.joint1_id].inverse().action
        @ data.oMi[contact_model.joint2_id].action
        @ pin.getJointJacobian(model, data, contact_model.joint2_id, pin.LOCAL)
    )
    J = c1_J_j1 - c1_J_j2

    return J


def _getConstraintJacobian(model, data, contact_model, contact_data):
    if contact_model.type == pin.ContactType.CONTACT_6D:
        return _getConstraintJacobian6d(model, data, contact_model, contact_data)
    elif contact_model.type == pin.ContactType.CONTACT_3D:
        return _getConstraintJacobian3d(model, data, contact_model, contact_data)
    else:
        assert False and "That's the two only possible contact types."


def _getConstraintsJacobian(model, data, constraint_models, constraint_datas):
    nc = len(constraint_models)
    assert len(constraint_datas) == nc
    Js = []
    for cm, cd in zip(constraint_models, constraint_datas):
        Js.append(_getConstraintJacobian(model, data, cm, cd))
    return np.vstack(Js)


if not P3X:
    pin.getConstraintJacobian = _getConstraintJacobian
    pin.getConstraintsJacobian = _getConstraintsJacobian

# -------------------------------------------------------------------------------
# Monkey patch of computeMinverse


def _computeMinverse(model: pin.Model, data: pin.Data, q: np.ndarray = None):
    """
    Monkey patch of computeMinverse.
    This is to allow a similar optional argument for q in p2x.
    Now you can write:  pin.computeMinverse(model,data,None if P3X else q).
    """
    assert q is not None or P3X
    if q is None:
        return pin._computeMinverse(model, data)
    else:
        return pin._computeMinverse(model, data, q)


pin._computeMinverse = pin.computeMinverse
pin.computeMinverse = _computeMinverse
