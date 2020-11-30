using DiffEqEnvironments
using SafeTestsets

@safetestset "DiffEqEnvironments.jl" begin
    #@safetestset "Random policy" begin include("random_policy.jl") end
    @safetestset "Scalar ODE" begin include("scalar_ode.jl") end
end