"""Base classes for Demagnetisation Problems"""

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"
        
from dolfin import *
from finmag.util.interiorboundary import InteriorBoundary
      
class TruncDemagProblem(object):
    """Base class for demag problems with truncated domains"""
    def __init__(self,mesh,subdomain,M,Ms = 1):
        """
        mesh - is the problem mesh
        
        subdomain- Subdomain an object of type SubDomain which defines an
                   inside of the mesh (to mark the magnetic region)

        M - the initial magnetisation (Expression at the moment)
        Ms - the saturation magnetisation
        """
        
        #(Currently M is constant)
        self.mesh = mesh
        self.M = M
        self.Ms = Ms
        self.subdomain = subdomain
        self.calculate_subsandbounds()

    def calculate_subsandbounds(self):
        """Calulate the submeshs and their common boundary"""
        
        #Mesh Function
        self.corefunc = MeshFunction("uint", self.mesh, self.mesh.topology().dim())
        self.corefunc.set_all(0)
        self.subdomain.mark(self.corefunc,1)
        
        #generate submesh for the core and vacuum
        self.coremesh = SubMesh(self.mesh,self.corefunc,1)
        self.vacmesh = SubMesh(self.mesh,self.corefunc,0)

        #generate interior boundary
        self.corebound = InteriorBoundary(self.mesh)
        self.corebound.create_boundary(self.coremesh)
        self.coreboundfunc = self.corebound.boundaries[0]

        #Store Value of coreboundary number as a constant
        self.COREBOUNDNUM = 2

        #generate measures
        self.dxC = dx(1)  #Core
        self.dxV = dx(0)  #Vacuum
        self.dSC = dS(self.COREBOUNDNUM)  #Core Boundary

    def refine_core(self):
        """Refine the Mesh inside the Magnetic Core"""
        #Mark the cells in the core
        cell_markers = CellFunction("bool", self.mesh)
        cell_markers.set_all(False)
        self.subdomain.mark(cell_markers,True)
        #Refine
        self.mesh = refine(self.mesh, cell_markers)
        #Regenerate Subdomains and boundaries
        self.calculate_subsandbounds()
