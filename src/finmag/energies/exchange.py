import numpy as np
import dolfin as df
import logging
from finmag.util.timings import timings
from energy_base import EnergyBase

logger = logging.getLogger('finmag')


class Exchange(EnergyBase):
    """
    Compute the exchange field.

    .. math::
        
        E_{\\text{exch}} = \\int_\\Omega A (\\nabla M)^2  dx
        
    *Arguments*
        C 
            the exchange constant
        method
            possible methods are 
                * 'box-assemble' 
                * 'box-matrix-numpy' 
                * 'box-matrix-petsc' [Default]
                * 'project'

    At the moment, we think (all) 'box' methods work 
    (and the method is used in Magpar and Nmag).

    - 'box-assemble' is a slower version that assembles the H_ex for a given M in every
      iteration.

    - 'box-matrix-numpy' precomputes a matrix g, so that H_ex = g*M

    - 'box-matrix-petsc' is the same mathematical scheme as 'box-matrix-numpy',
      but uses a PETSc linear algebra backend that supports sparse
      matrices, to exploit the sparsity of g (default choice).

    - 'project': does not use the box method but 'properly projects' the exchange field
      into the function space. Should explore whether this works and/or makes any difference
      (other than being slow.) Untested.


    *Example of Usage*

        .. code-block:: python

            m    = 1e-8
            n    = 5
            mesh = Box(0, m, 0, m, 0, m, n, n, n)
        
            S3  = VectorFunctionSpace(mesh, "Lagrange", 1)
            C  = 1.3e-11 # J/m exchange constant
            M  = project(Constant((Ms, 0, 0)), V) # Initial magnetisation

            exchange = Exchange(C, Ms)
            exchange.setup(S3, M)

            # Print energy
            print exchange.compute_energy()

            # Exchange field 
            H_exch = exchange.compute_field()

            # Using 'box-matrix-numpy' method (fastest for small matrices)
            exchange_np = Exchange(V, M, C, Ms, method='box-matrix-numpy')
            H_exch_np = exchange_np.compute_field()
            
    """
    def __init__(self, C, method="box-matrix-petsc"):
        logger.debug("Creating Exchange object with method {}.".format(method))
        self.in_jacobian = True
        self.C = C
        self.method = method
      
    def setup(self, S3, M, Ms, unit_length=1):
        timings.start('Exchange-setup')

        self.S3 = S3
        self.M = M  # keep reference to M
        self.Ms = Ms
        self.unit_length = unit_length

        self.mu0 = 4 * np.pi * 10 ** -7  # Vs/(Am)
        self.exchange_factor = df.Constant(1 * self.C / (self.mu0 * self.Ms * self.unit_length ** 2))

        self.v = df.TestFunction(S3)
        self.E = self.exchange_factor * df.inner(df.grad(M), df.grad(M)) * df.dx
        self.dE_dM = -1 * df.derivative(self.E, M, self.v)
        self.vol = df.assemble(df.dot(self.v, df.Constant([1, 1, 1])) * df.dx).array()
        self.dim = S3.mesh().topology().dim()

        # Needed for energy density
        S1 = df.FunctionSpace(S3.mesh(), "CG", 1)
        w = df.TestFunction(S1)
        self.nodal_vol = df.assemble(w * df.dx, mesh=S3.mesh()).array() \
                * unit_length ** self.dim
        self.nodal_E = df.dot(self.exchange_factor \
                * df.inner(df.grad(M), df.grad(M)), w) * df.dx

        # This is only needed if we want the energy density
        # as a df.Function, in order to e.g. probe.
        self.ED = df.Function(S1)

        # Don't know if this is needed
        self.total_vol = df.assemble(df.Constant(1) * df.dx, mesh=S3.mesh()) \
                * unit_length ** self.dim

        if self.method == 'box-assemble':
            self.__compute_field = self.__compute_field_assemble
        elif self.method == 'box-matrix-numpy':
            self.__setup_field_numpy()
            self.__compute_field = self.__compute_field_numpy
        elif self.method == 'box-matrix-petsc':
            self.__setup_field_petsc()
            self.__compute_field = self.__compute_field_petsc
        elif self.method == 'project':
            self.__setup_field_project()
            self.__compute_field = self.__compute_field_project
        else:
            print "Desired method was {}.".format(self.method)
            raise NotImplementedError("""Only methods currently implemented are
                                    * 'box-assemble',
                                    * 'box-matrix-numpy',
                                    * 'box-matrix-petsc'
                                    * 'project'""")

        timings.stop('Exchange-setup')

    def compute_field(self):
        """
        Compute the exchange field.
        
         *Returns*
            numpy.ndarray
                The exchange field.
        
        """
        timings.start('Exchange-computefield')
        Hex = self.__compute_field()
        timings.stop('Exchange-computefield')
        return Hex

    def compute_energy(self):
        """
        Return the exchange energy.

        *Returns*
            Float
                The exchange energy.

        """
        timings.start('Exchange-energy')
        E = df.assemble(self.E) * self.unit_length ** self.dim * self.Ms * self.mu0
        timings.stop('Exchange-energy')
        return E

    def energy_density(self):
        """
        Compute the exchange energy density,

        .. math::

            \\frac{E_\\mathrm{exch}}{V},

        where V is the volume of each node.

        *Returns*
            numpy.ndarray
                The exchange energy density.

        """
        nodal_E = df.assemble(self.nodal_E).array() \
                * self.unit_length ** self.dim * self.Ms * self.mu0
        return nodal_E / self.nodal_vol

    def energy_density_function(self):
        """
        Compute the exchange energy density the same way as the
        function above, but return a Function to allow probing.

        *Returns*
            dolfin.Function
                The exchange energy density.

        """
        self.ED.vector()[:] = self.energy_density()
        return self.ED

    def __setup_field_numpy(self):
        """
        Linearise dE_dM with respect to M. As we know this is
        linear ( should add reference to Werner Scholz paper and
        relevant equation for g), this creates the right matrix
        to compute dE_dM later as dE_dM=g*M.  We essentially
        compute a Taylor series of the energy in M, and know that
        the first two terms (dE_dM=Hex, and ddE_dMdM=g) are the
        only finite ones as we know the expression for the
        energy.

        """
        g_form = df.derivative(self.dE_dM, self.M)
        self.g = df.assemble(g_form).array()  # store matrix as numpy array

    def __setup_field_petsc(self):
        """
        Same as __setup_field_numpy but with a petsc backend.

        """
        g_form = df.derivative(self.dE_dM, self.M)
        self.g_petsc = df.PETScMatrix()

        df.assemble(g_form, tensor=self.g_petsc)
        self.H_ex_petsc = df.PETScVector()

    def __setup_field_project(self):
        #Note that we could make this 'project' method faster by computing the matrices
        #that represent a and L, and only to solve the matrix system in 'compute_field'().
        #IF this method is actually useful, we can do that. HF 16 Feb 2012
        H_exch_trial = df.TrialFunction(self.V)
        self.a = df.dot(H_exch_trial, self.v) * df.dx
        self.L = self.dE_dM
        self.H_exch_project = df.Function(self.V)

    def __compute_field_assemble(self):
        return df.assemble(self.dE_dM).array() / self.vol

    def __compute_field_numpy(self):
        Mvec = self.M.vector().array()
        H_ex = np.dot(self.g, Mvec)
        return H_ex / self.vol

    def __compute_field_petsc(self):
        self.g_petsc.mult(self.M.vector(), self.H_ex_petsc)
        return self.H_ex_petsc.array() / self.vol

    def __compute_field_project(self):
        df.solve(self.a == self.L, self.H_exch_project)
        return self.H_exch_project.vector().array()
