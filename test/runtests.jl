using DiffEqEnvironments
using Test
using SafeTestsets

@time @safetestset "DiffEqEnvironments.jl" begin
    @time @safetestset "Scalar ODE" begin include("scalar_autonomous_ode.jl") end
    @time @safetestset "LQR control" begin include("lqr_control.jl") end
    @time @safetestset "LTIQuadraticEnv" begin include("lti_constructor.jl") end
end