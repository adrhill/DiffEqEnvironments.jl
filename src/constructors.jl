"""
Constructs LTI system with quadratic reward and uniformly sampled initial conditions.
Key-word arguments match those of `DiffEqEnv`.
"""
function LTIQuadraticEnv(
    A::AbstractVecOrMat,
    B::AbstractVecOrMat,
    C::AbstractVecOrMat,
    D::AbstractVecOrMat,
    Q::AbstractVecOrMat,
    R::AbstractVecOrMat,
    s0_lb::Union{Real,Vector{<:Real}},
    s0_ub::Union{Real,Vector{<:Real}},
    tspan::Tuple{<:Real,<:Real},
    dt::Real;
    kwargs...
)
    n_states, n_actions, _ = state_space_validation(A, B, C, D, Continuous())
    ic_sampler = UniformSampler(s0_lb, s0_ub)

    # Sample random IC for dimension check and to define ODEProblem
    s0 = ic_sampler()
    length(s0) == n_states || throw(ArgumentError(
        "Length $(length(s0)) of s0 doesn't match state dimension $(n_states)"
    ))

    ode(s, a, t) = A * s + B * a # linear time-invariant ODE
    problem = ODEProblem(ode, s0, tspan)
    reward_fn = QuadraticReward(Q, R)
    observation_fn = LinearObservation(C, D)

    return DiffEqEnv(problem, reward_fn, n_actions, dt, s0_lb, s0_ub; kwargs...)
end

"""
Constructs LTI system with quadratic reward and constant initial conditions.
Key-word arguments match those of `DiffEqEnv`.
"""
function LTIQuadraticEnv(
    A::AbstractVecOrMat,
    B::AbstractVecOrMat,
    C::AbstractVecOrMat,
    D::AbstractVecOrMat,
    Q::AbstractVecOrMat,
    R::AbstractVecOrMat,
    s0::Union{Real,Vector{<:Real}},
    tspan::Tuple{<:Real,<:Real},
    dt::Real;
    kwargs...
)
    s0_lb = s0
    s0_ub = s0

    return LTIQuadraticEnv(A, B, C, D, Q, R, s0_lb, s0_ub, tspan, dt; kwargs...)
end
