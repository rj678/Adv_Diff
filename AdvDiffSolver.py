#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 12:35:40 2017

@author: rishi

General Advection Diffusion solver

Develop an API with the help of Python classes

Classes in this code:
    class AdvDiffSolver(object) - solver class
    class AdvDiffSuper(object) - super class
    

This code is based on the concepts and examples in:
    
Langtangen, Hans Petter and Logg, Anders. Solving PDEs in minutes -the FEniCS tutorial
Volume I. Springer, 2016.

and 

Langtangen, Hans Petter and Logg, Anders. Writing more advanced FEniCS programs -
the FEniCS tutorial Volume II. Springer, in prep.

"""

from fenics import *
import numpy as np





""" Create Solver class based on the theta rule and 
        the variational formulation """
        
class AdvDiffSolver(object):
    """ Solve an Adv-Diff problem by the theta rule """
    def __init__(self, problem, theta=0.5):
        self.problem = problem
        self.theta = theta
    
    def solve(self):
        """ Run time loop """
        tol = 1E-14
        T = self.problem.end_time()
        t = self.problem.time_step(0)
        self.initial_condtion()
        timestep = 1
        
        while t <= T+tol:
            # Solve variational form at next time step
            self.step(t)
            # Post-process the solution (compute errors)
            self.problem.user_action(t, self.u, timestep)
            
            # Update time step
            self.dt = self.problem.time_step(t+self.problem.time_step(t))
            t += self.dt
            # Assign solution value at previous time step
            self.u_1.assign(self.u)
            
    def initial_condtion(self):
        self.mesh, degree = self.problem.mesh_degree()
        self.V = V = FunctionSpace(self.mesh, 'P', degree)
        self.u_1 = interpolate(self.problem.IC(), V)
        self.u_1.rename('u', 'initial condition')
        self.u = self.u_1
        self.problem.user_action(0, self.u_1, 0)

    def step(self, t, linear_solver='direct',
                 abs_tol=1E-6, rel_tol=1E-5, max_iter=1000):
        """Advance solution one time step."""
        # Find new Dirichlet conditions at this time step
        Dirichlet_cond = self.problem.Dirichlet(t)
        if isinstance(Dirichlet_cond, Expression):
            # Just one Expression for Dirichlet bc on the entire boundary
            self.bcs = [DirichletBC(self.V, Dirichlet_cond,
                         lambda x, on_boundary: on_boundary)]
        else:
            # Boundary SubDomain markers
            self.bcs = [DirichletBC(self.V, value, boundaries, index)
                            for value, boundaries, index
                                in Dirichlet_cond]
        self.define_variational_problem(t)
        # Create bilinear and linear forms
        a, L = lhs(self.F), rhs(self.F)
        A = assemble(a)
        b = assemble(L)
        # Apply boundary conditions
        [bc.apply(A, b) for bc in self.bcs]
        
        # Solve linear system
        if linear_solver == 'direct':
            solve(A, self.u.vector(), b)
        else:
            solver = KrylovSolver('gmres', 'ilu')
            solver.solve(A, self.u.vector(), b)
    
    def define_variational_problem(self, t):
        """Set up variational problem at time t."""
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        # Declare parameters of problem
        dt = self.problem.time_step(t)
        kappa = self.problem.diff_coef()
        w = self.problem.adv_vel()
        rho = self.problem.density()
        f = self.problem.source(t)
        f_1 = self.problem.source(t-dt)
                        
        theta = Constant(self.theta)
        u_n = self.u_1 # computed in initial_condition
        
        # Declare variational form as in the notes
        F = rho*(u - u_n)/dt*v
        F += theta*dot(kappa*grad(u), grad(v))
        F += theta*dot(w,grad(u))*v
        F += (1-theta)*dot(kappa*grad(u_n), grad(v))
        F += (1-theta)*dot(w,grad(u))*v
        F -= theta*f*v + (1-theta)*f_1*v
        F = F*dx
        F += theta*sum([g*v*ds_ for g, ds_ in
                        self.problem.Neumann(t)])
        F += (1-theta)*sum([g*v*ds_ for g, ds_ in
                 self.problem.Neumann(t-dt)])
        F += theta*sum([r*(u - U_s)*v*ds_ for r, U_s, ds_ in
                        self.problem.Robin(t)])
        F += (1-theta)*sum([r*(u - U_s)*v*ds_ for r, U_s, ds_ in
                  self.problem.Robin(t-dt)])
        self.F = F
        
        # Make sure u is Function before solving
        self.u = Function(self.V)
        self.u.rename('u', 'solution')
                


# Declare Super class 
class AdvDiffSuper(object):
    """Abstract base class for Advection Diffusion problems"""
    def __init__(self, problem, debug=False):
        self.problem = problem
        self.debug = debug
    
    # Call colver class and return solution
    def solve(self, solver_class=AdvDiffSolver, theta=0.5, 
                  linear_solver='direct', abs_tol=1E-6,
                      rel_tol=1E-5, max_iter=1000):
        """Solve the PDE for the primary unknown"""
        self.solver = solver_class(self, theta)
        # Declare paramters for iterative solver
        iterative_solver = KrylovSolver('gmres', 'ilu')
        prm = iterative_solver.parameters
#        prm = parameters['krylov solver']
        prm['absolute_tolerance'] = abs_tol
        prm['relative_tolerance'] = rel_tol
        prm['maximum_iterations'] = max_iter
        prm['nonzero_initial_guess'] = True
        return self.solver.solve()
        
    def mesh_degree(self):
        """Return mesh, degree."""
        raise NotImplementedError('Must implement mesh')
        
    def IC(self):
        """ Return Initial condition """
        return Constant(0.0)
    
    def diff_coef(self): 
        return Constant(1.0)
    
    def adv_vel(self):
        # Make this work for 2D or 3D
        return Constant((0.0, 0.0))

    def density(self):
        return Constant(1.0)
    
    def source(self, t):
        return Constant(0.0)
    
    def time_step(self, t):
        raise NotImplementedError('Must implement end_time')
        
    def solution(self):
        return self.solver.u
    
    def user_action(self, t, u):
        """ Post process solution at time t """
        pass
    
    def Dirichlet(self, t):
        """ Return either an Expression for the entire boundary,
                or list of (value, boundary_parts, index) triplets """
        return []
    
    def Neumann(self, t):
        """ Return list of (g, ds(n)) pairs """
        return []
    
    def Robin(self, t):
        """ Return list of (r, s ds(n)) triplets """
        return []
    

























