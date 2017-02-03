from fenics import *
import numpy as np

def Newton_manual(J, F, u, bcs, atol, rtol, max_it, relax, u_res):
    # Reset counters
    Iter = 0
    residual = 1
    rel_res = residual

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
        print Iter
        if Iter == 0:
                rel_res0 = residual
                rel_res = 1
            else:
                rel_res = residual / rel_res0


        if MPI.rank(mpi_comm_world()) == 0:
            print "Newton iteration %d: r (atol) = %.3e (tol = %.3e), r (rel) = %.3e (tol = %.3e) " \
                            % (Iter, residual, atol, residual/rel_res0, rtol)
        Iter += 1
