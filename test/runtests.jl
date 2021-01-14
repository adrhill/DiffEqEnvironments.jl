using DiffEqEnvironments
using Test
using SafeTestsets

@time @safetestset "DiffEqEnvironments.jl" begin
    @time @safetestset "Scalar ODE" begin
        include("test_scalar_autonomous_ode.jl")
    end
    @time @safetestset "LQR control" begin
        include("test_lqr_control.jl")
    end
    @time @safetestset "LTIQuadraticEnv" begin
        include("test_lti.jl")
    end
    # @time @safetestset "MultiThreadEnv" begin
    #     include("test_multithread.jl")
    # end
    @time @safetestset "IC sampler" begin
        include("test_ic_sampler.jl")
    end
end
