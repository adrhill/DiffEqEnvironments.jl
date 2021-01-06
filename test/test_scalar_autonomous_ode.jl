using DiffEqEnvironments
using DifferentialEquations
using ReinforcementLearningBase
using ReinforcementLearningBase: test_interfaces!, test_runnable!
using Test

"""
This test defines an autonomous system and checks
whether the integration and terminal condition work
when stepping through the environment.
"""

# Define ODEProblem of scalar autonomous system
ode(s,a,t) = 1.01 * s
s0 = 1 / 2
tspan = (1f0, 2f0)
problem = ODEProblem(ode, s0, tspan)

# Solve ODE Problem for ground truth
sol = solve(problem, Tsit5())

# Define DiffEqEnv
r = ASReward((a, s) -> sum(abs2, s))
n_actions = 1
dt = 0.1
env = DiffEqEnv(problem, r, n_actions, dt; T=Float32, solver=Tsit5())

#= ==== Testing starts here ==== =#
# Test initialization
@test env.t == problem.tspan[1]
@test state(env) == s0

# Step through env until done
while !env.done
    env(0f0) # action doesn't matter for autonomous system
end

@test env.steps == (tspan[2] - tspan[1]) / dt
@test env.t ≈ problem.tspan[2]
@test state(env) ≈ sol.u[end]

#= ==== Reset and try again ==== =#
# Test initialization
RLBase.reset!(env)
@test env.t == problem.tspan[1]
@test state(env) == s0

# Step through env until done
while !env.done
    env(0f0) # action doesn't matter for autonomous system
end

@test env.steps == (tspan[2] - tspan[1]) / dt
@test env.t ≈ problem.tspan[2]
@test state(env) ≈ sol.u[end]

#= = Remake env using different types T = =#
env64 = DiffEqEnv(problem, r, n_actions, dt, T=Float64)
@test state(env64) == s0

env16 = DiffEqEnv(problem, r, n_actions, dt, T=Float16)
@test state(env16) == s0

# Test whether necessary interfaces from RLBase are implemented correctly and consistently
test_interfaces!(env)
test_runnable!(env)
