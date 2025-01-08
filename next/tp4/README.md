# Authorship

This set of examples and exercices comes from 3 sources:
- The supaero 2023 class
- The AWS class of Louis Montaut about collision detection
- The AWS class of Quentin Le Lidec about unilateral simulation

See https://github.com/Gepetto/supaero2023 and https://github.com/agimus-project/winter-school-2023 for the original files.

# Pinocchio 2x/3x compatibility

There are currently two major branches in both Pinocchio and HPP-FCL. The current versions are Pinocchio 2.7 and HPP-FCL 2.4.
Future release of HPP-FCL and Pinocchio are announced but not yet available.
For HPP-FCL, the devel branch has made significant progress but is not yet versionned. We will refer to this branch as the HPP-FCL 3x version.
For Pinocchio, there is a private repository with a version 3, but not yet publicly available. A pre-release is public, tagged 2.99 with a similar API but several uncorrected bugs (we don't recommande using it). We will refer to this branch as Pinocchio 3.
By Opposition, HPP-FCL 2.4 and Pinocchio 2.7 will be refered as 2x.


This set of exercices has been written to be compatible for both 2x and 3x versions. They should work with both.
A compatility.py file is introduced to force the behavior of 2x to match 3x.
Below is a list of major differences.

## Major compatibility breaks

### HPP-FCL 2x disfonctional cases

HPP-FCL 2x is producing several erroneous outputs that have been corrected in 3x.
This goes from not providing all information (sometime, normal is set to nan) to more critical failures (e.g. box-box collision is wrong, produces false witness points).
Also the normal direction have been inverted between 2x and 3x.

On the opposite, the distance function seems to behave more similarly between 2x and 3x.
We then force the collision function to be a simple post-process of the distance function.
The main drawback is that a single collision point is always found, contrary to what should be expected when calling the proper collision function.

Depending on the collision pairs, an ad-hoc post-process is applied to correct the output:
- sphere-sphere: the normal is recomputed from center-to-center segment
- sphere-plane: the normal is computed from the plane normal. Sanity check of the witness points.
- box-plane: the normal is computed from the plane normal. Sanity check of the witness points.
- box-box: the algorithm is incorrect, an assert is placed to prevent its use.

### computeMinverse
P3x allows evaluation as post-process of forward kinematics. The signature in 2x is changed to mimic it.

### Meshcat
Some primitives (ellipse, capsule) are now available in 2x. The *loadPrimitive* function is rewritten.

### RigidConstraint model and data
These classes are not available in 2x and are reimplemented in Python following the same API.
*getConstraintsJacobian* are reimplemented.

### Z axis
pin.ZAxis is not available in 2x and is reimplemented in Python.
