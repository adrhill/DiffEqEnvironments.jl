"""
Constructs LTI system with quadratic reward and uniformly sampled initial conditions.
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
    #= Keyword arguments =#
    o_ub::Union{Nothing,Real,Vector{<:Real}}=nothing, # upper bound for observation space
    o_lb::Union{Nothing,Real,Vector{<:Real}}=nothing, # lower bound for observation space
    a_ub::Union{Nothing,Real,Vector{<:Real}}=nothing, # upper bound for action space
    a_lb::Union{Nothing,Real,Vector{<:Real}}=nothing, # lower bound for action space
    solver::DiffEqBase.AbstractODEAlgorithm=Tsit5(), 
    reltol::Real=1e-8, 
    abstol::Real=1e-8, # TODO: add more kwargs for integrator
    T=Float32
    )

    n_states, n_actions, _ = state_space_validation(A, B, C, D, Continuous())
    ic_sampler = UniformSampler(s0_lb, s0_ub)

    # Sample random IC for dimension check and to define ODEProblem 
    s0 = ic_sampler()
    length(s0) == n_states || throw(ArgumentError("Length $(length(s0)) of s0 doesn't match state dimension $(n_states)")) 

    function ode(s, a, t) 
        ṡ = A * s + B * a
    end

    problem = ODEProblem(ode, s0, tspan)
    reward_fn = QuadraticReward(Q, R)
    observation_fn = LinearObservation(C, D)

    return DiffEqEnv(problem, reward_fn, n_actions, dt, s0_lb, s0_ub;  
        observation_fn=observation_fn,
        o_ub=o_ub, o_lb=o_lb, a_ub=a_ub, a_lb=a_lb,
        solver=solver, reltol=reltol, abstol=abstol,
        T=T
    )
end

"""
Constructs LTI system with quadratic reward and constant initial conditions.
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
    #= Keyword arguments =#
    o_ub::Union{Nothing,Real,Vector{<:Real}}=nothing, # upper bound for observation space
    o_lb::Union{Nothing,Real,Vector{<:Real}}=nothing, # lower bound for observation space
    a_ub::Union{Nothing,Real,Vector{<:Real}}=nothing, # upper bound for action space
    a_lb::Union{Nothing,Real,Vector{<:Real}}=nothing, # lower bound for action space
    solver::DiffEqBase.AbstractODEAlgorithm=Tsit5(), 
    reltol::Real=1e-8, 
    abstol::Real=1e-8, # TODO: add more kwargs for integrator
    T=Float32
    )

    s0_lb = s0
    s0_ub = s0

    return LTIQuadraticEnv(A, B, C, D, Q, R, 
        s0_lb, s0_ub, tspan, dt;
        o_ub=o_ub, o_lb=o_lb, a_ub=a_ub, a_lb=a_lb,
        solver=solver, reltol=reltol, abstol=abstol, T=T)
end