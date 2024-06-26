import logging
import dolfin as df
import numpy as np
from aeon import timer
from energy_base import EnergyBase
from finmag.field import Field
from finmag.util import helpers
from finmag.util.consts import mu0
from finmag.native import llg as native_llg
from finmag.field import Field

logger = logging.getLogger('finmag')


class UniaxialAnisotropyCylindrical(EnergyBase):
    """
    Compute the uniaxial anisotropy field.

    .. math::

        E_{\\text{anisc}} = \\int_\\Omega K_1 (1 - \\hat{r} \\cdot m)^2  dx

    *Arguments*
        K1
            The anisotropy constant
        K2
            The anisotropy constant (default=0)
        Ms
            The saturation magnetisation.
        method
            The method used to compute the anisotropy field.
            For alternatives and explanation, see EnergyBase class.

    *Example of Usage*
        .. code-block:: python

            import dolfin as df
            from finmag import UniaxialAnisotropy

            L = 1e-8;
            nL = 5;
            mesh = df.BoxMesh(df.Point(0, L, 0), df.Point(L, 0, L), nL, nL, nL)

            S3 = df.VectorFunctionSpace(mesh, 'Lagrange', 1)
            K = 520e3 # For Co (J/m3)

            a = df.Constant((0, 0, 1)) # Easy axis in z-direction
            m = df.project(df.Constant((1, 0, 0)), V)  # Initial magnetisation
            Ms = 1e6

            anisotropy = UniaxialAnisotropy(K, a)
            anisotropy.setup(S3, m)

            # Print energy
            print anisotropy.compute_energy()

            # Assign anisotropy field
            H_ani = anisotropy.compute_field()

    """

    def __init__(self, K1, K2=0, method="box-matrix-petsc", name='AnisotropyCylindrical', assemble=True):
        """
        Define a uniaxial anisotropy with (first) anisotropy constant `K1`
        (in J/m^3) and easy axis `axis`.

        K1 and axis can be passed as df.Constant or df.Function, although
        automatic convertion will be attempted from float for K1 and a
        sequence type for axis. It is possible to specify spatially
        varying anisotropy by using df.Functions.

        """
        self.K1_value = K1
        self.K2_value = K2
        # self.axis_value = axis
        self.name = name
        self.assemble = assemble
        super(UniaxialAnisotropyCylindrical, self).__init__(method, in_jacobian=True)

        if K2 != 0:
            self.assemble = False

    @timer.method
    def setup(self, m, Ms, unit_length=1):
        """
        Function to be called after the energy object has been constructed.

        *Arguments*

            m
                magnetisation field (usually normalised)

            Ms
                Saturation magnetisation field

            unit_length
                real length of 1 unit in the mesh

        """
        assert isinstance(m, Field)
        assert isinstance(Ms, Field)

        cg_scalar_functionspace = df.FunctionSpace(m.mesh(), 'CG', 1)
        self.K1 = Field(cg_scalar_functionspace, self.K1_value, name='K1')
        self.K2 = Field(cg_scalar_functionspace, self.K2_value, name='K2')

        cg_vector_functionspace = df.VectorFunctionSpace(m.mesh(), 'CG', 1, 3)
        self.axis = Field(cg_vector_functionspace, df.Expression(('cos(atan2(x[1], x[0]))', 'sin(atan2(x[1], x[0]))', '0'), degree=3), name='axis')

        # Anisotropy energy
        # HF's version inline with nmag, breaks comparison with analytical
        # solution in the energy density test for anisotropy, as this uses
        # the Scholz-Magpar method. Should anyway be an easy fix when we
        # decide on method.
        # FIXME: we should use DG0 space here?
        
        E_integrand = self.K1.f * \
            (df.Constant(1) - (df.dot(self.axis.f, m.f)) ** 2)
        if self.K2_value != 0:
            E_integrand -= self.K2.f * df.dot(self.axis.f, m.f) ** 4

        del(self.K1_value)
        del(self.K2_value)

        super(UniaxialAnisotropyCylindrical, self).setup(E_integrand, m, Ms, unit_length)

        if not self.assemble:
            self.H = self.m.get_numpy_array_debug()
            self.Ms = self.Ms.get_numpy_array_debug()
            self.u = self.axis.get_numpy_array_debug()
            self.K1_arr = self.K1.get_numpy_array_debug()
            self.K2_arr = self.K2.get_numpy_array_debug()
            self.volumes = df.assemble(df.TestFunction(cg_scalar_functionspace) * df.dx)
            self.compute_field = self.__compute_field_directly

    def __compute_field_directly(self):

        m = self.m.get_numpy_array_debug()

        m.shape = (3, -1)
        self.H.shape = (3, -1)
        self.u.shape = (3, -1)
        native_llg.compute_anisotropy_field(
            m, self.Ms, self.H, self.u, self.K1_arr, self.K2_arr)
        m.shape = (-1,)
        self.H.shape = (-1,)
        self.u.shape = (-1,)

        return self.H
