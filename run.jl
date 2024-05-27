# based on https://jso.dev/tutorials/advanced-jsosolvers/

using JSOSolvers
using CUTEst

using Ipopt
using NLPModelsIpopt

include("jso_manopt.jl")

cutest_problems = (
    CUTEstModel(p) for
    p in CUTEst.select(; objtype = "other", contype = "bounds", max_var = 100)
)


solvers = Dict(
    :tron => nlp -> tron(nlp, atol = 1.0e-4, rtol = 1.0e-4, max_time = 10.0),
    :ipopt => nlp -> ipopt(nlp, print_level = 0, sb = "no"),
    :manopt => nlp -> manopt_qn(nlp),
)

using SolverBenchmark
stats = bmark_solvers(solvers, cutest_problems)


costnames = ["time"]
costs = [df -> df.elapsed_time]

using Plots
gr()

profile_solvers(stats, costs, costnames)
