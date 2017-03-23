from fenics import *

def Newton_manual(J, F, u, u_res, bcs=[], atol=1e-12, rtol=1e-12, max_it=20,
                  relax=1, report_convergence=False):
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
        solve(A, u_res.vector(), solver_parameters={"linear_solver": "gmres"})

        # Update solution
        u.vector().axpy(relax, u_res.vector())
        [bc.apply(u.vector()) for bc in bcs]

        # Compute residual
        residual = b.norm('l2')
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
