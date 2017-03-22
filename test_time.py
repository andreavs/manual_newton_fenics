from fenics import *

mesh = UnitIntervalMesh(10)
x = SpatialCoordinate(mesh)

# Function space
W = FunctionSpace(mesh, "CG", 1)

# Define functions
u = TrialFunction(W)
v = TestFunction(W)
u_sol = Function(W)
t = Constant(0)

phi_cc = ((sin(pi*x[0]))**2 - 0.5)*cos(t)**2
phi_e = Expression('(pow(sin(pi*x[0]),2) - 0.5)*pow(cos(t),2)', t=0.)


a = u*v*dx
L = phi_cc*v*dx

tv = 0
dt = 0.1
for i in range(10):
    tv += dt
    t.assign(tv)
    phi_e.t = tv
    solve(a==L, u_sol)

print u_sol.vector().array()
phi_e_func = project(phi_e, W)
print phi_e_func.vector().array()
error = u_sol.vector().array() - phi_e_func.vector().array()
print error
