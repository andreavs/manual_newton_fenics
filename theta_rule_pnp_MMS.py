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

set_log_level(99)

def run_mms(dt, N, end_time, theta=0):
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

    theta = Constant(theta)
    c1_theta = theta*c1 + (1-theta)*c1_new
    c2_theta = theta*c2 + (1-theta)*c2_new
    phi_theta = theta*phi + (1-theta)*phi_new
    dummy_theta = theta*d + (1-theta)*dummy_new

    # Params:
    F = 9.648e4   # Faradays constant, C/mol
    T = 300       # Temperature, Kelvin
    R = 8.314     # Rayleighs constant, J/(K*mol)
    psi = R*T/F
    eps_0 = 8.854 # Vacuum permitivity, pF/m
    eps_r = 80    # Relative permitivity of water, no dimension
    eps = eps_0*eps_r

    D1 = 2.0 # diffusion coefficient
    D2 = 1.0 # diffusion coefficient
    z1 = 1   # valency
    z2 = -1  # valency

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
    k = Constant(dt)
    form = ((c1_new-c1)*v_1 + k*inner(D1*nabla_grad(c1_theta) + \
        D1*c1_theta*z1*nabla_grad(phi_theta)/psi, nabla_grad(v_1)) - k*f1*v_1)*dx + \
        ((c2_new-c2)*v_2 + k*inner(D2*nabla_grad(c2_theta) + \
        D2*c2_theta*z2*nabla_grad(phi_theta)/psi, nabla_grad(v_2)) - k*f2*v_2)*dx + \
        (eps*inner(nabla_grad(phi_new), nabla_grad(v_phi)) + dummy_new*v_phi + phi_new*d - rho*v_phi)*dx

    dw = TrialFunction(W)
    Jac = derivative(form, u_new, dw)
    u_res = Function(W)

    tv = 0
    n_iter = int(end_time / dt)
    for i in range(n_iter):
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
    error_c2 = errornorm(c2_e, u.sub(1), norm_type="l2", degree_rise=3)
    error_phi = errornorm(phi_e, u.sub(2), norm_type="l2", degree_rise=3)

    return error_c1, error_c2, error_phi, mesh.hmin()


def run_convergence(N_list, dt_list):
    errors_c1 = []
    errors_c2 = []
    errors_phi = []
    h = []
    type_of_convergence = "Spatial" if len(N_list) > len(dt_list) else "Temporal"
    print "="*15, type_of_convergence, "="*15
    end_time = dt_list[0]*10 if type_of_convergence == "Spatial" else max(dt_list)*4

    for N in N_list:
        for dt in dt_list:
            print N, dt
            error_c1, error_c2, error_phi, hmin = run_mms(dt, N, end_time,
                                                          theta=0.5)
            h.append(hmin)
            errors_c1.append(error_c1)
            errors_c2.append(error_c2)
            errors_phi.append(error_phi)

    N = max(len(N_list), len(dt_list)) - 1
    h = h if len(N_list) > len(dt_list) else dt_list

    print "Spatial convergence C1"
    for i in range(N):
        print math.log(errors_c1[i] / errors_c1[i+1]) / math.log(h[i] / h[i+1])

    print "Spatial convergence C2"
    for i in range(N):
        print math.log(errors_c2[i] / errors_c2[i+1]) / math.log(h[i] / h[i+1])

    print "Spatial convergence phi"
    for i in range(N):
        print math.log(errors_c2[i] / errors_c2[i+1]) / math.log(h[i] / h[i+1])

    print "\n"

if __name__ == '__main__':
        run_convergence([100, 200, 400, 800], [1e-6])
        run_convergence([800], [0.001, 0.0005, 0.00025])
