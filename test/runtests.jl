using DiffEqEnvironments
using Test
using SafeTestsets

@time @safetestset "DiffEqEnvironments.jl" begin
    @time @safetestset "Scalar ODE" begin include("test_scalar_autonomous_ode.jl") end
    @time @safetestset "LQR control" begin include("test_lqr_control.jl") end
    @time @safetestset "LTIQuadraticEnv" begin include("test_lti.jl") end
end