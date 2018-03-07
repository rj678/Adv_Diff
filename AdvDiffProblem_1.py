#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 13:18:44 2017

@author: rishi

Different problems are implemented as different 
subclasses of AdvDiffProblem (found in AdvDiffSolver.py)


Classes in this code:
    class Example_1(ad.AdvDiffSuper)

This code is based on the concepts and examples in:
    
Langtangen, Hans Petter and Logg, Anders. Solving PDEs in minutes -the FEniCS tutorial
Volume I. Springer, 2016.

and 

Langtangen, Hans Petter and Logg, Anders. Writing more advanced FEniCS programs -
the FEniCS tutorial Volume II. Springer, in prep.

"""

# Import the required modules
from fenics import *
import numpy as np
import AdvDiffSolver as ad




""" Test Problem: 
    Pure Diffusion Problem
    Manufactured Solution is u = 1 + x^2 + alpha*y^2 + beta*t """
    
class Example_1(ad.AdvDiffSuper):
    """ Specific Instance of Super class AdvDiffSuper    """
    def __init__(self, Nx, Ny, Nz=None, degree=1, num_time_steps=5):
        if Nz is None:
            self.mesh = UnitSquareMesh(Nx, Ny)
        else:
            self.mesh = UnitCubeMesh(Nx, Ny, Nz)
        self.degree = degree
        self.num_time_steps = num_time_steps
        
        # Declare coeff in exact solution
        alpha = 3
        beta = 1.2
        
        # Initial Condition (t=0)
        self.u0 = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                        degree=degree, alpha=alpha, beta=beta, t=0)
        # Make this part simpler and make it work if vel is a function
        w = self.adv_vel()
        vals = np.zeros(2)
        w.eval(vals, np.zeros(2))
        wx = vals[0]
        wy = vals[1]
        # Declare Source term
        self.f = Expression('beta - 2 - 2*alpha + 2*x[0]*wx + 2*alpha*x[1]*wy',
                                degree=degree, alpha=alpha, beta=beta,
                                    wx = wx, wy = wy)
    
    # Declare problem specific functions    
    def time_step(self, t):
        return 0.1
    
    def end_time(self):
        return self.num_time_steps*self.time_step(0)
    
    def mesh_degree(self):
        return self.mesh, self.degree
    
    def IC(self):
        """Return initial condition."""
        return self.u0
    
    def source(self, t):
        return self.f
    
    def adv_vel(self):
        return Constant((2.0, 2.0))
    
    def Dirichlet(self, t):
        self.u0.t = t
        return self.u0
    
#    def Neumann(self):
#        """Return list of g*ds(n) values."""
#        return [(0, self.ds(0)), (0, self.ds(1))]
#    
    def user_action(self, t, u, timestep):
        """This for post-processing the solution u at time t."""
        # Interpolate exact solution onto the function space
        u_e = interpolate(self.u0, u.function_space())
        error = np.abs(u_e.vector().array() -
                       u.vector().array()).max()
        print('error at %g: %g' % (t, error))
        tol = 1E-10
        assert error < tol, 'max_error: %g' % error


""" Unit test: """
def test_DiffusionSolver():
#if __name__ == '__main__':
    problem = Example_1(Nx=2, Ny=2)
    # Solve the PDE using the specified solver - Direct or Krylov (GMRES-ILU)
    # Specify theta - BE(1), CN(0.5) or FE(0)
    problem.solve(theta=1.0, linear_solver='direct')
    #problem.solve(theta=1, linear_solver='Krylov')
    #u = problem.solution()
















