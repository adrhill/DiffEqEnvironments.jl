using DiffEqEnvironments
using DifferentialEquations
using LinearAlgebra
using ControlSystems
using Test
using ReinforcementLearningBase
using Plots

"""
This test defines an SISO system of form
    ṡ = A*s + B*a
    o = C*s
for the rocket car problem, where
    s = [p, ṗ]  ,
p being the position of the car.
The action used as an input to the system is 
the car's acceleration p̈ ∈ [-1, 1]. Therefore
    ṡ = [ṗ, p̈] = [0 1; 0 0] * s + [0; 1] * u .
Full observation of the state is given, as
    o = s   .
"""
T = Float32

# System matrices
A = [0 1; 0 0]
B = [0, 1]

# weights for quadratic cost function
Q = I * 10
R = I

# Set range of control input
a_lb = -1.0; a_ub = 1.0

# Set IC & tspan for integration
s0 = [-0.75, 0.0]
tspan = (0f0, 5f0)

# Define ODEProblem of SISO LTI system
function lti(s, a, t) 
    ṡ = A * s + B * a
end
prob_lti = ODEProblem(lti, s0, tspan)

# Use ControlSystems.jl to solve continuous algebraic Riccati eq.
P = care(A, B, Q, R) # P encodes value function as v(s)= -s'*P*s/2
K = -R \ B' * P
println("K=", K)

v_lqr(s) = -s' * P * s / 2
π_lqr(s) = clamp.(K * s, a_lb, a_ub) #  LQR controller w/ saturation


# Define closed-loop ODE for system with input saturations
function cl_ode(s, π, t)
    a = π(s) 
    ṡ = lti(s, a, t)
end

function plot_value(v; levels=10)
    ps = range(-1, 1, length=100)
    ṗs = range(-1, 1, length=100)
    vp(p, ṗ) = v([p,ṗ]) 

    contour(ps, ṗs, vp, levels=levels,
        xlabel="p", ylabel="ṗ",
        fill=true, show=true)
end

p = plot_value(v_lqr)

# Plot LQR Solution
prob_lqr = ODEProblem(cl_ode, s0, tspan, π_lqr)
sol_lqr = solve(prob_lqr, Tsit5())

function plot_phase_portrait!(plt, π; n=20, scale=0.075)
    ṡ(s) = lti(s, π(s), 0) * scale
    ṡ(p, ṗ) = ṡ([p,ṗ])

    ps = range(-1, 1, length=n)
    ṗs = range(-1, 1, length=n)

    pps = [p for p in ps for ṗ in ṗs]
    ṗṗs = [ṗ for p in ps for ṗ in ṗs]
    quiver!(pps, ṗṗs, quiver=ṡ, lw=1, c=:black,
        xlim=(-1, 1), ylim=(-1, 1),
        arrow=(style = :closed; headlength = 0.1))
end

function plot_trajectory!(plt, s0, π; c=:black)
    prob = ODEProblem(cl_ode, s0, tspan, π)
    sol = solve(prob, Tsit5(), saveat=0.01)

    plot!(plt, sol[1,:], sol[2,:], lw=3, c=c,
        xlim=(-1, 1), ylim=(-1, 1), line=:arrow,
        legend=false, show=true)
end

p2 = plot_phase_portrait!(p, π_lqr)
p3 = plot_trajectory!(p, s0, π_lqr)

display(p)

# Define DiffEqEnv
r = QuadraticRewardFunction(Q, R)
n_actions = 1
dt = 0.1
env = DiffEqEnv(prob_lti, r, n_actions, dt, a_lb=a_lb, a_ub=a_ub)