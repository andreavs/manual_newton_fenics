from fenics import *
import numpy as np
import time

def Newton_manual(J, F, u, u_res, bcs=[], atol=1e-12, rtol=1e-12, max_it=20,
                  relax=1, report_convergence=True, c1_e=None, V=None):
    # Reset counters
    Iter = 0
    residual = 1
    rel_res = residual
    f = Function(V)
    f2 = Function(V)
    while rel_res > rtol and residual > atol and Iter < max_it:
        # Assemble system
        A = assemble(J)
        b = assemble(-F)

        # Solve linear system
        [bc.apply(A, b, u.vector()) for bc in bcs]
        solve(A, u_res.vector(), b)

        # Update solution
        u.vector().axpy(relax, u_res.vector())
        [bc.apply(u.vector()) for bc in bcs]

        # Compute residual
        residual = b.norm('l2')
        ba = b.array()
        ba = ba[1:np.size(f2.vector().array())+1]
        f2.vector()[:] = ba
        print np.size(b.array())
        print np.size(f.vector().array())
        c1 = u.sub(0)
        assign(f, c1)
        f.assign(f-project(c1_e,V))
        plot(f, title="Iteration:" + str(Iter))
        plot(f2, title="residual")
        time.sleep(0.5)

        if Iter == 0:
            rel_res0 = residual
            rel_res = 1
        else:
            rel_res = residual / rel_res0


        if MPI.rank(mpi_comm_world()) == 0:
            if report_convergence:
                print "Newton iteration %d: r (atol) = %.3e (tol = %.3e), r (rel) = %.3e (tol = %.3e) " \
                                % (Iter, residual, atol, rel_res, rtol)
        Iter += 1
