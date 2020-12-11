using DiffEqEnvironments
using DifferentialEquations
using ReinforcementLearning
using Test

"""
This test defines an autonomous system and checks
whether the integration and terminal condition work
when stepping through the environment.
"""

# Define ODEProblem of scalar autonomous system
ode(s,a,t) = 1.01*s
s0 = 1/2
tspan = (1f0, 2f0)
problem = ODEProblem(ode, s0, tspan)

# Solve ODE Problem for ground truth
sol = solve(problem, Tsit5())

# Define DiffEqEnv
r = ASReward((a,s) -> sum(abs2,s))
n_actions = 1
dt = 0.1
env = DiffEqEnv(problem, r, n_actions, dt; T=Float32, solver=Tsit5())

#===== Testing starts here =====#
# Test initialization
@test env.t == problem.tspan[1]
@test env.state[1] == s0 

# Step through env until done
while !env.done
    env(0f0) # action doesn't matter for autonomous system
end

@test env.steps == (tspan[2]-tspan[1])/dt
@test env.t ≈ problem.tspan[2]
@test env.state[1] ≈ sol.u[end]

#===== Reset and try again =====#
# Test initialization
RLBase.reset!(env)
@test env.t == problem.tspan[1]
@test env.state[1] == s0 

# Step through env until done
while !env.done
    env(0f0) # action doesn't matter for autonomous system
end

@test env.steps == (tspan[2]-tspan[1])/dt
@test env.t ≈ problem.tspan[2]
@test env.state[1] ≈ sol.u[end]


#== Remake env using different types T ==#
env64 = DiffEqEnv(problem, r, n_actions, dt, T=Float64)
@test env64.state[1] == s0 

env16 = DiffEqEnv(problem, r, n_actions, dt, T=Float16)
@test env16.state[1] == s0 

#== Run random policy using ReinforcementLearning.jl ==#
RLBase.reset!(env)
hook = TotalRewardPerEpisode()
run(
    Agent(
        ;policy = RandomPolicy(env),
        trajectory = VectCompactSARTSATrajectory(
            state_type=Any,
            action_type=Any,
            reward_type=Real,
            terminal_type=Bool,
        ),
    ),
    env,
    StopAfterEpisode(10),
    hook
)