using DiffEqEnvironments
using SafeTestsets

@safetestset "DiffEqEnvironments.jl" begin
    @safetestset "Scalar ODE" begin include("scalar_autonomous_ode.jl") end
    @safetestset "LQR control" begin include("lqr_control.jl") end
end