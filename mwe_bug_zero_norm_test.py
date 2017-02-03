from fenics import *
from newton import Newton_manual

mesh = UnitSquareMesh(10,10)
V = FunctionSpace(mesh, 'CG', 1)
v = TestFunction(V)
u = Function(V)

def boundary(x, on_boundary):
    return on_boundary

bcs = [DirichletBC(V, Constant(0), boundary)]
F = inner(nabla_grad(u), nabla_grad(v))*dx
dw = TrialFunction(V)
Jac = derivative(F, u, dw)
tol = 1e-12
relax = 1.0
u_res = Function(V)
Newton_manual(Jac, F, u, bcs, tol, tol, 20, relax, u_res)
