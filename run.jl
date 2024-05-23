using Revise
using LBFGSB
using Optim
using Manopt
using CUTEst, NLPModels

function run_lbfgsb(
    nlp::CUTEstModel;
    mem_size::Int = 10,
    pgtol::Float64 = 1e-5,
    maxfun::Int = 15000,
    maxiter::Int = 15000,
    factr::Float64 = 1e7,
)
    n = nlp.meta.nvar
    optimizer = L_BFGS_B(n, mem_size)
    x = nlp.meta.x0
    # set up bounds
    bounds = zeros(3, n)
    bounds[1, :] .= 2
    # bounds[1, i] represents the type of bounds imposed on the variables:
    #  0->unbounded, 1->only lower bound, 2-> both lower and upper bounds, 3->only upper bound
    bounds[2, :] .= nlp.meta.lvar
    bounds[3, :] .= nlp.meta.uvar

    # 
    f(x) = obj(nlp, x)
    g!(z, x) = grad!(nlp, x, z)
    optimizer(f, g!, x, bounds; m = mem_size, factr, pgtol, iprint = -1, maxfun, maxiter)
    t0 = time()
    fout, xout = optimizer(
        f,
        g!,
        x,
        bounds;
        m = mem_size,
        factr,
        pgtol,
        iprint = -1,
        maxfun,
        maxiter,
    )
    t1 = time()

    return fout, t1 - t0
end

struct AlgConfig{TF,TKW}
    fun::TF
    kwargs::TKW
end

function run()
    # box constraints
    problems = CUTEst.select(; objtype = "other", contype = "bounds")
    println("Selected $(length(problems)) problems")

    alg_configs =
        AlgConfig[AlgConfig(run_lbfgsb, (;)), AlgConfig(run_lbfgsb, (; mem_size = 5))]

    foms = zeros(length(problems), length(alg_configs))

    for (ip, prob) in enumerate(problems)
        nlp = CUTEstModel(prob)
        println("Problem $(prob)")
        println("Minimize: $(nlp.meta.minimize)")
        println("size(x0) = $(size(nlp.meta.x0))")
        println("fx = $( obj(nlp, nlp.meta.x0) )")

        for (iac, ac) in enumerate(alg_configs)
            fout, time = ac.fun(nlp; ac.kwargs...)
            foms[ip, iac] = time
        end

        finalize(nlp)
    end

    println(foms)
end

run()
