from fenics import *
from newton import Newton_manual
import math

"""
In this script we aim to solve the Poisson-Nernst-Planck equations, i.e. the
concentration dynamics of two ions, assuming they act by electrodiffusion.

Variables are
c1 - concentration of ion type 1
c2 - concentration of ion type 2
phi - the electric field

The dynamics equations are
dc1/dt = D(nabla^2(c1) + (1/psi)*nabla(c1*nabla(phi)))
dc2/dt = D(nabla^2(c1) + (1/psi)*nabla(c1*nabla(phi)))
nabla^2 phi = F(z1*c1 + z2*c2)/eps

D, psi, F, eps are physical constants (see below).
z1, z2 are the valencies of the ions.

In this example, the ions start with a high concentration on the left half of
an interval, and a low concentration on the right half. The initial
concentrations are equal.

We use dirichlet boundary conditions for the ions, and a pure von neuman
boundary for the electric field.
"""


def run_mms(dt,N, theta=1):
    mesh = UnitIntervalMesh(N)

    # Defining functions and FEniCS stuff:
    V = FunctionSpace(mesh, 'CG', 1)
    R = FunctionSpace(mesh, 'R', 0)
    W = MixedFunctionSpace([V, V, V, R])
    v_1, v_2, v_phi, d = TestFunctions(W)

    time = 0
    u = Function(W)
    u_new = Function(W)

    c1, c2, phi, dummy = split(u)
    c1_new, c2_new, phi_new, dummy_new = split(u_new)

    c1_theta = (1-theta)*c1 + theta*c1_new
    c2_theta = (1-theta)*c2 + theta*c2_new
    phi_theta = (1-theta)*phi + theta*phi_new
    dummy_theta = (1-theta)*d + theta*dummy_new

    # Params:
    F = 9.648e4 # Faradays constant, C/mol
    T = 300 # Temperature, Kelvin
    R = 8.314 # Rayleighs constant, J/(K*mol)
    psi = R*T/F
    eps_0 = 8.854 # Vacuum permitivity, pF/m
    eps_r = 80 # Relative permitivity of water, no dimension
    eps = eps_0*eps_r

    D1 = 2.0 # diffusion coefficient
    D2 = 1.0 # diffusion coefficient
    z1 = 1 # valency
    z2 = -1 # valency

    # dt = 1e-3 # time step, ms


    t = Constant(0)
    x = SpatialCoordinate(mesh)

    phi_cc = ((sin(pi*x[0]))**2 - 0.5)*cos(t)**2
    c1_cc = cos(x[0])**3 * sin(t)
    c2_cc = 1/z2*(-eps/F*div(nabla_grad(phi_cc)) - z1*(c1_cc))


    f1 = diff(c1_cc,t) - D1*div(nabla_grad(c1_cc) + (1.0/psi)*z1*c1_cc*nabla_grad(phi_cc))
    f2 = diff(c2_cc,t) - D2*div(nabla_grad(c2_cc) + (1.0/psi)*z2*c2_cc*nabla_grad(phi_cc))



    phi_e = Expression("(pow(sin(pi*x[0]), 2) - 0.5) * pow(cos(t),2)", t=time, degree=4)
    c1_e = Expression("pow(cos(x[0]), 3) * sin(t)", D1=D1, t=time, degree=4)
    c2_e = Expression("1.0/z2*(-eps/F*2*pi*pi*pow(cos(t),2)*cos(2*pi*x[0]) - z1*pow(cos(x[0]), 3) * sin(t))",  \
        z2=z2, z1=z1, eps=eps, F=F, degree=4, t=time)

    assign(u.sub(0), interpolate(c1_e, V))
    assign(u.sub(1), interpolate(c2_e, V))
    assign(u.sub(2), interpolate(phi_e, V))

    # boundary conditions
    def boundary(x, on_boundary):
        return on_boundary

    bcs = [DirichletBC(W.sub(0), c1_e, boundary), DirichletBC(W.sub(1), c2_e, boundary)]

    rho = F*(z1*c1_theta + z2*c2_theta)
    form = ((c1_new-c1)*v_1 + dt*inner(D1*nabla_grad(c1_theta) + \
        D1*c1_theta*z1*nabla_grad(phi_theta)/psi, nabla_grad(v_1)) - dt*f1*v_1)*dx + \
        ((c2_new-c2)*v_2 + dt*inner(D2*nabla_grad(c2_theta) + \
        D2*c2_theta*z2*nabla_grad(phi_theta)/psi, nabla_grad(v_2)) - dt*f2*v_2)*dx + \
        (eps*inner(nabla_grad(phi_new),nabla_grad(v_phi)) + dummy_new*v_phi + phi_new*d - rho*v_phi)*dx

    dw = TrialFunction(W)
    Jac = derivative(form, u_new, dw)
    u_res = Function(W)


    tv = 0

    N_t = int(0.01/dt)

    error_c1 = Function(V)
    error_c2 = Function(V)
    for i in range(100):
        tv += dt
        c1_e.t = tv
        c2_e.t = tv
        phi_e.t = tv
        t.assign(tv)

        Newton_manual(Jac, form, u_new, u_res,bcs=bcs, max_it=1000, atol = 1e-9, rtol=1e-8)
        assign(u, u_new)

    c1_e_f = project(c1_e, V)
    c1_sol = Function(V)
    assign(c1_sol, u.sub(0))

    error_c1 = errornorm(c1_e, c1_sol, norm_type="l2", degree_rise=3)
    # plot(project(c1_e,V))
    # plot(u.sub(0))
    # interactive()
    error_c2 = errornorm(c2_e, u.sub(1), norm_type="l2", degree_rise=3)
    error_phi = errornorm(phi_e, u.sub(2), norm_type="l2", degree_rise=3)
    return error_c1, error_c2, error_phi, mesh.hmin()

if __name__ == '__main__':
        errors_c1 = []
        errors_c2 = []
        errors_phi = []
        h = []
        N_list = [100, 200, 400, 800]
        dt = 1e-7

        # Spatial convergence test
        for N in N_list:
            error_c1, error_c2, error_phi, hmin = run_mms(dt, N, theta=1)
            h.append(hmin)
            errors_c1.append(error_c1)
            errors_c2.append(error_c2)
            errors_phi.append(error_phi)
            dt = dt

        print h
        print errors_c1
        print errors_c2
        print errors_phi
        for i in range(len(N_list) - 1):
            print math.log(errors_c1[i] / errors_c1[i+1]) / math.log(h[i] / h[i+1])
        for i in range(len(N_list) - 1):
            print math.log(errors_c2[i] / errors_c2[i+1]) / math.log(h[i] / h[i+1])
        for i in range(len(N_list) - 1):
            print math.log(errors_phi[i] / errors_phi[i+1]) / math.log(h[i] / h[i+1])


        # print "Spatial convergence rate:"
        # for i in range(len(N_list) - 1):
        #     print math.log(error[i] / error[i+1]) / math.log(h[i] / h[i+1])
