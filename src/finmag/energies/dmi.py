import logging
import dolfin as df
from aeon import timer
from finmag.field import Field
from energy_base import EnergyBase
from finmag.util.helpers import times_curl
import numpy as np

logger = logging.getLogger('finmag')


def cyl_rho(r):
    """Cylindrical unit vector rho
    
    The unit vector is defined as a vector field::
    .. math::
        \\hat{\\rho} = \\cos(\\phi) \\hat{x} + \\sin(\\phi) \\hat{y}
    """
    phi = np.arctan2(r[1], r[0])
    return np.array([np.cos(phi), np.sin(phi), 0.0])


class DMI(EnergyBase):
    """
    Compute the Dzyaloshinskii-Moriya Interaction (DMI) field.
    .. math::
        E_{\\text{DMI}} = \\int_\\Omega D \\vec{m} \\cdot
                          (\\nabla \\times \\vec{m}) dx
    *Arguments*
        D
            the DMI constant
        method
            See documentation of EnergyBase class for details.
        dmi_type
            Options are 'auto', '1d', '2d', '3d' and 'interfacial'.
            Default value is 'auto' which means the dmi is automaticaly
            selected according to the mesh dimension.
    *Example of Usage*
        .. code-block:: python
            import dolfin as df
            from finmag.energies.dmi import DMI
            from finmag.field import Field
            # Define a mesh representing a cube with edge length L.
            L = 1e-8  # m
            n = 5
            mesh = df.BoxMesh(df.Point(0, L, 0), df.Point(L, 0, L), n, n, n)
            D = 5e-3  # J/m**2 DMI constant
            Ms = 0.8e6  # A/m magnetisation saturation
            # Initial magnetisation
            S3 = df.VectorFunctionSpace(mesh, 'CG', 1)
            m = Field(S3, (1, 0, 0))
            dmi = DMI(D)
            dmi.setup(m, Ms)
            # Compute DMI energy.
            E_dmi = dmi.compute_energy()
            # Compute DMI effective field.
            H_dmi = dmi.compute_field()
            # Using 'box-matrix-numpy' method (fastest for small matrices)
            dmi_np = DMI(D, method='box-matrix-numpy')
            dmi_np.setup(m, Ms)
            H_dmi_np = dmi_np.compute_field()
    """

    def __init__(self, D, method='box-matrix-petsc', name='DMI', dmi_type='auto'):
        self.D_value = D  # Value of D, later converted to a Field object.
        self.name = name
        self.dmi_type = dmi_type

        super(DMI, self).__init__(method, in_jacobian=True)


    @timer.method
    def setup(self, m, Ms, unit_length=1):
        # Create an exchange constant Field object A in DG0 function space.
        dg_functionspace = df.FunctionSpace(m.mesh(), 'DG', 0)
        # self.D = Field(dg_functionspace, self.D_value, name='D')
        # del(self.D_value)

        # Coercing any scalar values into a list here.  Not too elegant, but
        # it simplifies the treatment in LI_assembler().  The (possible)
        # deprecation of DMI_interfacial will simplify it again.
        if type(self.D_value) == list:
            self.D = [ Field(dg_functionspace, x, name='D') for x in self.D_value ]
        else:
            self.D = [ Field(dg_functionspace, self.D_value, name='D') ]
        del(self.D_value)

        # Multiplication factor used for dmi energy computation.
        self.dmi_factor = df.Constant(1.0 / unit_length)

        if self.dmi_type == '1d':
            dmi_dim = 1
        elif self.dmi_type == '2d':
            dmi_dim = 2
        elif self.dmi_type == '3d':
            dmi_dim = 3
        else:
            dmi_dim = m.mesh_dim()

        # Select the right expression for computing the dmi energy.
        if self.dmi_type == 'interfacial':
            E_integrand = DMI_interfacial(m, self.dmi_factor*self.D[0].f, dim=dmi_dim)
        elif self.dmi_type == 'interfacial_cyl':
            # Equivalent: df.VectorFunctionSpace(e.mesh, "Lagrange", 1, dim=3)
            # S3 = df.VectorFunctionSpace(m.mesh(), 'CG', 1, 3)

            # S3 = df.VectorFunctionSpace(m.mesh(), 'CG', 1, 3)
            # rho_field = Field(S3)
            # rho_field.set(cyl_rho)
            # self.rho_field = df.interpolate(df.Expression(['cos(atan2(x[1], x[0]))', 'sin(atan2(x[1], x[0]))', '0.0'], degree=1), S3)
            
            # Lagrange field (Continuos Galerkin) with 1 dof per vertex, and continuous across cells
            # If we used 'DG', dim=0; we would have 1 dof per cell (centre of tetrahedra)
            S0 = df.FunctionSpace(m.mesh(), 'CG', 1)
            self.rho_field = []
            self.rho_field.append(df.interpolate(df.Expression('cos(atan2(x[1], x[0]))', degree=1), S0))
            self.rho_field.append(df.interpolate(df.Expression('sin(atan2(x[1], x[0]))', degree=1), S0))

            # self.rho_field = []
            # self.rho_field.append(Field(df.FunctionSpace(m.mesh(), 'DG', 1), df.Expression('cos(atan2(x[1], x[0]))', degree=1)))
            # self.rho_field.append(Field(df.FunctionSpace(m.mesh(), 'DG', 1), df.Expression('sin(atan2(x[1], x[0]))', degree=1)))
            
            E_integrand = DMI_interfacial_cylindrical(m, self.rho_field, self.dmi_factor*self.D[0].f)
        # elif self.dmi_type == 'D2D':
        #     E_integrand = DMI_D2D(m, self.dmi_factor*self.D.f, dim=dmi_dim)
        elif self.dmi_type in ['C_nv', 'D_n', 'D_2d', 'C_n', 'S_4']:
            D_fields = [ self.dmi_factor * x.f for x in self.D ]
            E_integrand = LI_assembler(m, D_fields, self.dmi_type)
        else:
            E_integrand = self.dmi_factor*self.D[0].f*times_curl(m.f, dmi_dim)

        super(DMI, self).setup(E_integrand, m, Ms, unit_length)

        if self.method == 'direct':
            self.__setup_field_direct()

    def __setup_field_direct(self):
        dofmap = self.m.mesh_dofmap()
        S3 = df.VectorFunctionSpace(self.m.mesh(), "CG", 1, dim=3,
                                   constrained_domain=dofmap.constrained_domain)

        u3 = df.TrialFunction(S3)
        v3 = df.TestFunction(S3)
        self.g_petsc = df.PETScMatrix()
        df.assemble(-2*self.dmi_factor*self.D.f*df.inner(v3, df.curl(u3))*df.dx, tensor=self.g_petsc)
        self.H_petsc = df.PETScVector()

def DMI_interfacial(m, D, dim):
    """
    Input arguments:
    m is a Field object on a 1d or 2d space,
    D is the DMI constant
    dim is the mesh dimension.
    Returns the form to compute the DMI energy:
    D(m_x * dm_z/dx - m_z * dm_x/dx) +
    D(m_y * dm_z/dy - m_z * dm_y/dy) * df.dx
    References:
    [1] Rohart, S. and Thiaville A., Phys. Rev. B 88, 184422 (2013)
    """

    gradm = df.grad(m.f)

    dmxdx = gradm[0, 0]
    dmydx = gradm[1, 0]
    dmzdx = gradm[2, 0]

    if dim == 1:
        dmxdy = 0
        dmydy = 0
        dmzdy = 0
    else:
        # Works for both 2d mesh or 3d meshes.
        dmxdy = gradm[0, 1]
        dmydy = gradm[1, 1]
        dmzdy = gradm[2, 1]

    mx = m.f[0]
    my = m.f[1]
    mz = m.f[2]

    return D*(mx * dmzdx - mz * dmxdx) + D*(my * dmzdy - mz * dmydy)


def DMI_interfacial_cylindrical(m, rho_field, D):
    """
    Interfacial DMI for cylindrical system with out-of-surface normal vector

    Only working in 3d for now
    """

    mx = m.f[0]
    my = m.f[1]
    mz = m.f[2]

    # mn = df.inner(rho_field, m.f)

    # S1 = df.FunctionSpace(m.mesh(), 'DG', 1)
    # mn = df.interpolate(df.Expression('mx * cos(atan2(x[1], x[0])) + my * sin(atan2(x[1], x[0]))', 
    #                                   mx=m.f.sub(0), my=m.f.sub(1), degree=1), S1)

    mr = rho_field[0] * mx + rho_field[1] * my
    # mn_grad = df.grad(mn)
    # m_dot_gradr = m.f[0] * mn_grad[0] + m.f[1] * mn_grad[1] + m.f[2] * mn_grad[2]
                                  #
    gradm = df.grad(m.f)
    dmxdx = gradm[0, 0]
    dmydx = gradm[1, 0]
    dmxdy = gradm[0, 1]
    dmydy = gradm[1, 1]
    dmxdz = gradm[0, 2]
    dmydz = gradm[1, 2]

    # gradm_x_rm = df.inner(mn, df.div(m.f))
    gradm_x_rm = mr * df.div(m.f)
    rho_dot_mgrad_m = (rho_field[0] * (mx * dmxdx + my * dmxdy + mz * dmxdz) +
                       rho_field[1] * (mx * dmydx + my * dmydy + mz * dmydz)) 

    return D * (gradm_x_rm - rho_dot_mgrad_m)


def DMI_D2D(m, D, dim):
    """
    D2D type DMI

    w = D (L_xz^y + L_yz^x)

    where L_ij^k = m_i dm_j / dk - m_j dm_i / dk

    Hence:

    w = D [(m_x dm_z / dy - m_z dm_x / dy) +
           (m_y dm_z / dx - m_z dm_y / dx)]

      = D (m_x dm_z/dy + m_y dm_z/dx - m_z ( dm_x / dy + dm_y / dx))

    """

    if dim != 3:
        raise ValueError("This DMI Type does not work other than in 3-D.")

    gradm = df.grad(m.f)

    dmxdy = gradm[0, 1]
    dmydx = gradm[1, 0]
    dmzdx = gradm[2, 0]
    dmzdy = gradm[2, 1]

    mx = m.f[0]
    my = m.f[1]
    mz = m.f[2]

    return D * (mx * dmzdy + my * dmzdx - mz * ( dmxdy + dmydx))


def LI_factory(m, d = [ 0, 0, 0, 0, 0, 0, 0, 0, 0 ]):
    """ Refer to LI_assembler() for a more in-depth explanation. """
    #
    # Collect the ingredients: 9 DMI constants (argument), magnetization
    # components and all (first) spatial derivatives thereof.
    #
    mx = m.f[0]
    my = m.f[1]
    mz = m.f[2]
    gradm = df.grad(m.f)
    dmxdx = gradm[0, 0]
    dmydx = gradm[1, 0]
    dmzdx = gradm[2, 0]
    dmxdy = gradm[0, 1]
    dmydy = gradm[1, 1]
    dmzdy = gradm[2, 1]
    dmxdz = gradm[0, 2]
    dmydz = gradm[1, 2]
    dmzdz = gradm[2, 2]

    #
    # Now built up the most general DMI term, 'total_dmi', which is composed
    # of 9 Lifshitz invariants and corresponding DMI constants.  Selection of
    # a particular crystal symmetry then reduces their number (drastically).
    #
    # The constants are named d[i] to slightly avoid headaches with the
    # 'D' variable passed to the class constructor.
    #
    # TODO Insert deep insight about the cross-product nature and the
    # resulting interpretation as regards crystal symmetry here.
    # Something along the lines of: a Lifshitz invariant,
    #   m_i \delta_\kappa m_j - m_j \delta_\kappa m_i
    # describes a spiral state for |epsilon_{i,j,\kappa}| = 1,
    # else, a helical state.
    total_dmi = \
    d[0]*(my*dmzdx - mz*dmydx) + \
    d[1]*(my*dmzdy - mz*dmydy) + \
    d[2]*(my*dmzdz - mz*dmydz) + \
    d[3]*(mz*dmxdx - mx*dmzdx) + \
    d[4]*(mz*dmxdy - mx*dmzdy) + \
    d[5]*(mz*dmxdz - mx*dmzdz) + \
    d[6]*(mx*dmydx - my*dmxdx) + \
    d[7]*(mx*dmydy - my*dmxdy) + \
    d[8]*(mx*dmydz - my*dmxdz)

    return total_dmi

def LI_assembler(m, D, crystalsym):
    """
    This is the LI assembler for the terms as listed in ANB89.

    The selection of DMI constants implies a specific crystal symmetry,
    yet here we /start/ with the crystal symetry (argument) and
    interpret the D's (argument, a list) accordingly.

    And remember: "To stabilize vortex states, an additional
    uniaxial anisotropy must be present." (Bogdanov 1994)
    """
    #
    # Here's a short overview of DMI terms mentioned in literature.
    #
    ### A.N. Bogdanov and D.A. Yablonsky, Sov. Phys. JETP 95 (1989) 178:
    #
    #   C_nv: a' w_1,              D_n: a'_1 w_2,             D_2d: a'_2 w_2',
    #   C_n: a'_3 w_1 + a'_4 w_2,  S_4: a'_5 w_1' + a'_6 w_2'
    #
    # where
    #
    #   a'_i are simply different DMI constants; and
    #   w_i are composed of one or more lifshitz invariants (given here
    #   directly as elements of total_dmi in LI_factory)
    #
    #   w_1  = d[3] - d[1],  w_2  = d[4] + d[0],
    #   w_1' = d[3] + d[1],  w_2' = d[4] - d[0]
    #   (w_3 = d[8])
    #
    # The d[i] refer to the terms in LI_factory (hence the array notation).
    #
    # Note, w_3 is mentioned, not treated:
    # > We note also that the classes D_n and C_n admit the invariants
    # > [w_3] which can lead to the formation of a spiral structure with
    # > propagation vector along the z axis.
    #
    ### A.N. Bogdanov and A. Hubert, JMMM 138, 255 (1994):
    #
    # General term
    #
    #   w_D = a_1 w_1 + a_2 w_2 + a_3 w_3
    #
    # which, translated to LI_factory terms, becomes
    #
    #       = a_1 ( d[3] - d[1] ) +
    #         a_2 ( d[4] + d[0] ) +
    #         a_3 ( d[8] )
    #
    # These expressions correspond to C_nv (a_2 = 0) and D_n (a_1 = 0) above.
    # The spiral structures in z direction (a_3 term), though explicitly
    # included, are not treated in this paper, either.
    #
    ### Rohart, S. and Thiaville A., Phys. Rev. B 88, 184422 (2013):
    #
    # The 'interface term' becomes, in this notation,
    #
    #   a_0 ( -d[3] + d[1] )
    #
    # which, in Bogdanov's parlance is simply -C_nv
    # (again, without the z-spiral, of course).

    # And now, for the Elegant Dispatch Table(TM).
    # TODO The dichotomy between d[] in LI_factory and D[] here is
    # confusing and the table is very far from elegant.  Ideas?
    #
    # Check requirements for the different point groups, then let
    # LI_factory construct the appropriate DMI term.
    if crystalsym in ['C_nv', 'D_n', 'D_2d']:  # Requirements...
        if not 1 <= len(D) <= 2:
            logger.warning("Cannot initialize '", crystalsym, "': strictly one or two DMI constants necessary.")
            return LI_factory(m) # TODO The empty interaction?  Rather: discard.
        if len(D) == 1:  # No z-term given.
            D.append(0)

        # DMI construction
        if crystalsym is 'C_nv':
            return LI_factory(m, [0, -D[0], 0, D[0], 0, 0, 0, 0, D[1]])
        elif crystalsym is 'D_n':
            return LI_factory(m, [D[0], 0, 0, 0, D[0], 0, 0, 0, D[1]])
        elif crystalsym is 'D_2d':
            return LI_factory(m, [-D[0], 0, 0, 0, D[0], 0, 0, 0, D[1]])

    elif crystalsym in ['C_n', 'S_4']:
        if not len(D) == 2:
            logger.warning("Cannot initialize '", crystalsym, "': strictly two DMI constants necessary.")
            return LI_factory(m)

        if crystalsym is 'C_n':
            return LI_factory(m, [D[1], -D[0], 0, D[0], D[1], 0, 0, 0, 0])
        elif crystalsym is 'S_4':
            return LI_factory(m, [-D[1], D[0], 0, D[0], D[1], 0, 0, 0, 0])

    else:
        logger.warning("Unknown crystal symmetry: ", crystalsym)
        return LI_factory(m)  # TODO Is this indeed a suitable no-op?
