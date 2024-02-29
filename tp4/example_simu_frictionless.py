import pinocchio as pin
import hppfcl
import numpy as np
from supaero2024.meshcat_viewer_wrapper import MeshcatVisualizer
from tp4.compatibility import P3X
from scenes import buildSceneCubes,addFloor
from display_collision_patches import preallocateVisualObjects,updateVisualObjects
from create_rigid_contact_models_for_hppfcl import createContactModelsFromCollisions
import matplotlib.pyplot as plt
import time
import proxsuite; QP = proxsuite.proxqp.dense.QP

# Parameters of the simulation
DURATION = 3. # duration of simulation
DT = 1e-3 # time step duration
DT_VISU = 1/100
T = int(DURATION/DT) # number of time steps
WITH_CUBE_CORNERS = True # Use contacts only at cube corners
PRIMAL_FORMULATION = True
DUAL_FORMULATION = True
assert(PRIMAL_FORMULATION or DUAL_FORMULATION)
# Random seed for simulation initialization
SEED = int(time.time()%1*1000); print('SEED = ',SEED)
SEED = 1

# ### RANDOM INIT
# Init random generators
np.random.seed(SEED)
pin.seed(SEED)

# ### SCENE
# Create scene with multiple objects
model,geom_model = buildSceneCubes(1,with_floor=True,with_corner_collisions=True,with_cube_collisions=False)

# Create the corresponding data to the models
data = model.createData()
geom_data = geom_model.createData()

for req in geom_data.collisionRequests:
    req.security_margin = 1e-3
    req.num_max_contacts = 20

# ### VIZUALIZATION
visual_model = geom_model.copy()
preallocateVisualObjects(visual_model,10)
viz = MeshcatVisualizer(model=model, collision_model=geom_model,
                        visual_model=visual_model)
updateVisualObjects(model,data,[],[],visual_model,viz)

# ### INIT MODEL STATE
q0 = model.referenceConfigurations['default']

viz.display(q0)

q = q0.copy()
v = np.zeros(model.nv)

# ### LOGS
hq = []
hv = []

# ### MAIN LOOP
# ### MAIN LOOP
# ### MAIN LOOP
for t in range(T):

    # Compute free dynamics
    tau = np.zeros(model.nv)
    pin.computeCollisions(model, data, geom_model, geom_data, q)
    vf = v + DT * pin.aba(model, data, q, v, tau)

    # Create contact models from collision
    contact_models = createContactModelsFromCollisions(model,data,geom_model,geom_data)
    contact_datas = [ cm.createData() for cm in contact_models ]
    
    nc = len(contact_models)
    if nc==0:
        # No collision, just integrate the free dynamics
        v = vf
    else:
        # With at least one collision ...
        # Compute mass matrix.
        # (pin.crba should be enough in P3X, but CAT to be sure in P2X);
        pin.computeAllTerms(model, data, q, v)
        # The contact solver express the constraint in term of velocity of
        # body 1 wrt to body 2. It is more intuitive to think to the opposite
        # so take the negative jacobian (ie velocity of body 2 wrt body 1, whose
        # normal component should be positive).
        # Select only normal components of contact
        # (frictionless slide on the tangent components, uncontrained)
        J = -pin.getConstraintsJacobian(model, data, contact_models, contact_datas)[2::3,:]
        assert(J.shape == (nc,model.nv))
        
        if PRIMAL_FORMULATION:
            # ### PRIMAL_FORMULATION:

            # Solve the primal QP (search the velocity)
            # min_v  .5 vMv - vfMv st Jv>=0
            qp1 = QP(model.nv,0,nc,False)
            qp1.init(data.M,-data.M@vf,[],[],J,l=np.zeros(nc))#,u=np.ones(nc)*1e20)
            qp1.settings.eps_abs = 1e-12
            qp1.solve()

            vnext = qp1.results.x
            # By convention, proxQP takes negative multipliers for the lower bounds
            # We prefer to see the positive forces (ie applied by body 1 to body 2).
            forces = -qp1.results.z

            # Check the solution respects the physics
            assert(np.all(forces>=-1e-6))
            assert(np.all(J@vnext>=-1e-6))
            assert(abs( forces@J@vnext)<1e-6)
            # Check the acceleration obtained from the forces
            assert( np.allclose(pin.aba(model, data, q, v, tau + J.T @ forces/DT),
                                (vnext-v)/DT,rtol=1e-3,atol=1e-6))

        if DUAL_FORMULATION:
            # ### DUAL FORMULATION
            # Solve the dual QP (search the forces)
            # min_f .5 f D f st f>=0, D f + J vf >= 0
            # Compatibility issue: the 3rd argument is to enforce compatibility between
            # pinocchio 2x and 3x. Remove it if with P3X, or replace it by q if with P2X.
            Minv = pin.computeMinverse(model,data,None if P3X else q)
            delasus = J@Minv@J.T

            qp2 = QP(nc,0,nc,box_constraints=True)
            qp2.settings.eps_abs = 1e-12
            # Both side of the box constraint must be given
            # otherwise the behavior of the solver is strange
            qp2.init(
                H=delasus,g=np.zeros(nc),
                C=delasus,l=-J@vf,
                l_box=np.zeros(nc),u_box=np.ones(nc)*np.inf
             )
            qp2.solve()

            # Compute the contact acceleration from the forces
            forces = qp2.results.x
            vnext = v + DT * pin.aba(model, data, q, v, tau + J.T @ forces/DT)

            # Check the solution respects the physics
            assert(np.all(forces>=-1e-6))
            assert(np.all(J@vnext>=-1e-6))
            assert(abs( forces@J@vnext)<1e-6)

        if PRIMAL_FORMULATION and DUAL_FORMULATION:
            # Check QP2 primal vs QP1 dual
            assert(np.allclose(qp2.results.x,-qp1.results.z,rtol=1e-3,atol=1e-4))
            # Check QP2 constraint vs QP1 constraint
            assert(np.allclose(delasus@qp2.results.x+J@vf,
                               J@qp1.results.x,rtol=1,atol=1e-5))
        
        v = vnext
        
    # Finally, integrate the valocity
    q = pin.integrate(model , q, v*DT)

    # Log
    hq += [q]
    hv += [v]

    # Visualize once in a while
    if DT_VISU is not None and abs((t*DT) % DT_VISU)<=0.9*DT:
        updateVisualObjects(model,data,contact_models,contact_datas,visual_model,viz)
        viz.display(q)
        time.sleep(DT_VISU)
