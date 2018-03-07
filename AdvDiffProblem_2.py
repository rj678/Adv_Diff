#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 12:49:42 2017

@author: rishijumani

Different problems are implemented as different 
subclasses of AdvDiffProblem (found in AdvDiffSolver.py)


Classes in this code:
    class Example_2(ad.AdvDiffSuper)

This code is based on the concepts and examples in:
    
Langtangen, Hans Petter and Logg, Anders. Solving PDEs in minutes -the FEniCS tutorial
Volume I. Springer, 2016.

and 

Langtangen, Hans Petter and Logg, Anders. Writing more advanced FEniCS programs -
the FEniCS tutorial Volume II. Springer, in prep.

"""

# Import the required modules
#from __future__ import print_function
from fenics import *
import numpy as np
import AdvDiffSolver as ad


import matplotlib.pyplot as plt


def mark_boundary(mesh, d=2, x0=0, x1=1, y0=0, y1=1, z0=0, z1=1):
	"""
	SubDomain and Expression Python classes can be used to define subdomains,
	but they lead to C++ function calls (to Python) for each node in the mesh,
	 which is very expensive
	
	You can use C++ code directly to specify the subdomain
	
	CompiledSubdomain can be used to specify boundaries as well
	
	Returns mesh function FacetFunction with each side in a hypercube
	of d dimensions.
	
	 Sides are marked by indicators 0, 1, 2, ..., 6.
	Side 0 is x=x0, 1 is x=x1, 2 is y=y0, 3 is y=y1, and so on. 
	"""
	
	sides = ['near(x[0], %(x0)s, tol)', 'near(x[0], %(x1)s, tol)',
					       'near(x[1], %(y0)s, tol)', 'near(x[1], %(y1)s, tol)',
					         'near(x[2], %(z0)s, tol)', 'near(x[2], %(z1)s, tol)']
	
	boundaries = [CompiledSubDomain(
			('on_boundary && ' + side) % vars(),  tol=1E-14)
				for side in sides[:2*d]]
	
	# Mark boundaries
	boundary_parts = FacetFunction('uint', mesh)
	boundary_parts.set_all(9999)
	for i in range(len(boundaries)):
		boundaries[i].mark(boundary_parts, i)
	return boundary_parts



""" Test Problem: 
    Development of Thermal Boundary Layer
     """
    
class Example_2(ad.AdvDiffSuper):
    """ Specific Instance of Super class AdvDiffSuper    """
    def __init__(self, Nx, Ny, Nz=None, degree=1, num_time_steps=5):
				 
        self.make_mesh(Nx, Ny, Nz)
        self.degree = degree		 
        self.num_time_steps = num_time_steps
        
        """ The advective part of the code and the initial condition have 
		    been commented out and not deleted  """
        # Declare coeff in exact solution
        #alpha = 3
        #beta = 1.2
        
        # Initial Condition (t=0)
        #self.u0 = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
         #               degree=degree, alpha=alpha, beta=beta, t=0)
        # Make this part simpler and make it work if vel is a function
#        w = self.adv_vel()
#        vals = np.zeros(2)
#        w.eval(vals, np.zeros(2))
#        wx = vals[0]
#        wy = vals[1]
        # Declare Source term
        #self.f = Expression('beta - 2 - 2*alpha + 2*x[0]*wx + 2*alpha*x[1]*wy',
         #                       degree=degree, alpha=alpha, beta=beta,
           #                         wx = wx, wy = wy)

	
    def make_mesh(self, Nx, Ny, Nz):
		""" Initialize mesh and mark boundaries """
		
		if Nz is None:
			self.mesh = UnitSquareMesh(Nx, Ny)
		else:
			self.mesh = UnitCubeMesh(Nx, Ny, Nz)
		
		
		self.boundary = mark_boundary(self.mesh, d=2)
		
		# Redefine the measure ds in terms of boundary markers 
		self.ds = Measure('ds', domain=self.mesh, 
					           subdomain_data=self.boundary)
		
		
	
    # Declare problem specific functions    
    def time_step(self, t):
        # Small time steps initially while the boundary layer develops
        
        if t < 0.02:
            return 0.0005
        else:
            return 0.025
		    
    def end_time(self):
        return 0.3
    
    def mesh_degree(self):
        return self.mesh, self.degree
    
    def IC(self):
        """Return initial condition."""
        return Constant(0.0)
    
    def source(self, t):
        return Constant(0.0)
    
    def adv_vel(self):
        return Constant((0.0, 0.0))
    
    def Dirichlet(self, t):
		""" Return list of (value, boundary) pairs"""
		
		return [(1.0, self.boundary, 0),
				  (0.0, self.boundary, 1)]               
    
#    def Neumann(self):
#        """Return list of g*ds(n) values."""
#        return [(0, self.ds(0)), (0, self.ds(1))]
#    
    def user_action(self, t, u, timestep):
        """For post-processing the solution u at time t"""
 
        tol = 1E-14
        
#       Evolution of solution plotted as curves in one plot 	
#         at y = 0.5

        x_vec = np.linspace(tol, 1-tol, 101)
        x = [(x_,0.5) for x_ in x_vec]
        u = self.solution()
        u_sol = [u(x_) for x_ in x]


        plt.figure(1)
        plt.plot(x_vec, u_sol, '-')
        plt.title('Evolution of Solution at y = 0.5')
        plt.legend(['Final time is t=%.4f' %t])
        plt.xlabel('x'); plt.ylabel('u')
        plt.axis([0, 1, 0, 1])
        plt.hold('on')


""" Unit test: """
#def test_DiffusionSolver():
if __name__ == '__main__':
    problem = Example_2(Nx=20, Ny=5)
    # Solve the PDE using the specified solver - Direct or Krylov (GMRES-ILU)
    # Specify theta - BE(1), CN(0.5) or FE(0)
    problem.solve(theta=1.0, linear_solver='direct')
    plt.figure(1)
    plt.savefig('tmp1.png'); plt.savefig('tmp1.pdf')
    #problem.solve(theta=1, linear_solver='Krylov')
    #u = problem.solution()
















