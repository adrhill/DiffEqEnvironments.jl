using DiffEqEnvironments
using DifferentialEquations
using Test

ode(s,a,t) = 1.01*s
s0 = 1/2
tspan = (0f0, 1f0)
problem = ODEProblem(ode, s0, tspan)

r = RewardFunction(s->s^2)
n_actions = 1
dt = 0.1

print(problem.u0)

env = DiffEqEnv(problem, r, n_actions, dt)

print(env)

@test env.state == [s0]


#ODEProblem(env.ode!, env.s0, tspan, p=action)