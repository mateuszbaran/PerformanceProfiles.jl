

using NLPModels, Manopt, SolverCore, Manifolds

using ImprovedHagerZhangLinesearch

"""
    ManoptSolver(nlp; kwargs...,)

Returns an `ManoptSolver` structure to solve the problem `nlp` with `ipopt`.
"""
mutable struct ManoptSolver <: AbstractOptimizationSolver
    state::AbstractManoptSolverState
end

function manopt_qn(nlp::AbstractNLPModel; kwargs...)
    M = Hyperrectangle(nlp.meta.lvar, nlp.meta.uvar)
    p0 = copy(nlp.meta.x0)
    du = QuasiNewtonLimitedMemoryDirectionUpdate(
        M,
        p0,
        InverseBFGS(),
        min(manifold_dimension(M), 10),
    )
    sc = StopAfterIteration(100) | StopWhenGradientNormLess(1e-6)
    state = QuasiNewtonState(
        M,
        p0;
        direction_update = du,
        stepsize = HagerZhangLinesearch(M),
        stopping_criterion = sc,
    )
    solver = ManoptSolver(state)
    stats = GenericExecutionStats(nlp)
    return SolverCore.solve!(solver, nlp, stats; kwargs...)
end

function get_status(state::AbstractManoptSolverState)
    asc = Manopt.get_active_stopping_criteria(state.stop)
    if Manopt.indicates_convergence(state.stop)
        return :first_order
    elseif any(sc -> isa(sc, StopAfterIteration), asc)
        return :max_iter
    elseif any(sc -> isa(sc, StopWhenStepsizeLess), asc)
        return :small_step
    else
        return :unknown
    end
end

function SolverCore.solve!(
    solver::ManoptSolver,
    nlp::AbstractNLPModel,
    stats::GenericExecutionStats;
    callback = (args...) -> true,
    kwargs...,
)
    reset!(stats)

    if !nlp.meta.minimize
        error("Maximization is not available at the moment")
    end
    M = Hyperrectangle(nlp.meta.lvar, nlp.meta.uvar)

    function eval_f(M::AbstractManifold, p)
        return obj(nlp, project(M, p))
    end
    function eval_grad_f(M::AbstractManifold, X, p)
        grad!(nlp, p, X)
        project!(M, X, p, X)
        #println(norm(X), "\t", obj(nlp, p), "\t|||", p)
        return X
    end

    gmp = ManifoldGradientObjective(eval_f, eval_grad_f; evaluation = InplaceEvaluation())
    mp = DefaultManoptProblem(M, gmp)

    real_time = time()
    Manopt.solve!(mp, solver.state)
    real_time = time() - real_time

    x_sol = get_solver_result(solver.state)

    dual_feas = Inf
    primal_feas = distance(M, x_sol, project(M, x_sol))
    iter = get_count(solver.state, :Iterations)

    set_status!(stats, get_status(solver.state))
    set_solution!(stats, x_sol)
    set_objective!(stats, obj(nlp, x_sol))
    set_residuals!(stats, primal_feas, dual_feas)
    set_iter!(stats, iter)
    set_time!(stats, real_time)
    return stats
end
