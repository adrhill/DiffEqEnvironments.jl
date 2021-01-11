"""
Struct holding parameters for ODE solver
"""
mutable struct DiffEqParams{T}
    problem::ODEProblem             # ODEProblem which is simulated
    dt::T                           # time step size
    solver::AbstractODEAlgorithm    # solver to use to solve ODE
    solve_args::Dict                # dictionary holding kwargs for ODE solve command
end

mutable struct DiffEqEnv{T,R<:AbstractRNG} <: AbstractEnv
    ode_params::DiffEqParams{T}
    ic_sampler::ICSampler
    # Parameters for ReinforcementLearning
    observation_space::Union{Interval,Space}
    action_space::Union{Interval,Space}
    # Observation & reward functions
    observation_fn::ObservationFunction     # obvervation o=f(s)
    reward_fn::RewardFunction               # reward function r(s,a,s')
    # Buffer for previous transition
    state::Union{T,Vector{T}}               # current state
    observation::Union{T,Vector{T}}         # current observation
    reward::T                               # last scalar reward
    done::Bool                              # true if in terminal state
    steps::Int                              # counter for steps taken in episode
    t::T                                    # current time
    rng::R
end

"""
User-facing constructor for DiffEqEnv with IC sampling
"""
function DiffEqEnv(
    problem::ODEProblem,
    reward_fn::RewardFunction,
    n_actions::Int,
    dt::Real,
    s0_lb::Union{Real,Vector{<:Real}},
    s0_ub::Union{Real,Vector{<:Real}};
    #= Keyword arguments =#
    observation_fn::ObservationFunction=FullStateObservation(),
    o_ub::Union{Nothing,Real,Vector{<:Real}}=nothing, # upper bound for observation space
    o_lb::Union{Nothing,Real,Vector{<:Real}}=nothing, # lower bound for observation space
    a_ub::Union{Nothing,Real,Vector{<:Real}}=nothing, # upper bound for action space
    a_lb::Union{Nothing,Real,Vector{<:Real}}=nothing, # lower bound for action space
    solver::DiffEqBase.AbstractODEAlgorithm=Euler(),
    reltol::Real=1e-8,
    abstol::Real=1e-8, # TODO: add more kwargs for integrator
    T=Float64,
    rng=Random.GLOBAL_RNG,
)
    # Save solver arguments
    solve_args = Dict{Symbol,Any}(
        :reltol => reltol,
        :abstol => abstol,
        :save_everystep => false,
        :save_start => false, # only output terminal value
    )

    ode_params = DiffEqParams{T}(problem, dt, solver, solve_args)

    # Set inital state and observation
    ic_sampler = UniformSampler(s0_lb, s0_ub; rng=rng)
    s0 = T.(ic_sampler()) # initial state
    o0 = T.(observation_fn(s0, nothing)) # initial observations

    n_states = length(s0)
    n_observations = length(o0)

    # Helper function to set action and observation spaces
    function set_space(lb, ub, dim)
        # Check if types match
        typeof(lb) == typeof(ub) ||
            throw(ArgumentError("$(typeof(lb)) != $(typeof(ub)), types must match"))

        # Set defaults if no bounds are provided
        if isnothing(lb)
            if dim == 1
                ub = T(1e38)
                lb = -ub
            else
                ub = T(1e38) * ones(T, dim)
                lb = -ub
            end
        end

        # Check if sizes match
        length(lb) == length(ub) ||
            throw(ArgumentError("$(size(lb)) != $(size(ub)), size must match"))
        length(lb) == dim || throw(ArgumentError("$(size(lb)) != $(dim), size must match"))

        if lb isa Real
            return ClosedInterval{T}(lb, ub)
        elseif lb isa Vector{<:Real}
            return Space(ClosedInterval{T}.(lb, ub))
        end
    end

    # Set observation and action spaces
    observation_space = set_space(o_lb, o_ub, n_observations)
    action_space = set_space(a_lb, a_ub, n_actions)

    # Initialize buffer holding previous step
    state = s0
    observation = o0
    reward = T(-Inf)
    done = false
    steps = 0
    t = problem.tspan[1]

    env = DiffEqEnv{T,typeof(rng)}(
        ode_params,
        ic_sampler,
        observation_space,
        action_space,
        observation_fn,
        reward_fn,
        state,
        observation,
        reward,
        done,
        steps,
        t,
        rng,
    )
    return env
end

"""
User-facing constructor for DiffEqEnv with constant ICs
"""
function DiffEqEnv(
    problem::ODEProblem,
    reward_fn::RewardFunction,
    n_actions::Int,
    dt::Real;
    T=Float64,
    kwargs...,
)

    # Set bounds for ICs to same values
    # This ensures that the same IC gets sampled by env.ic_sampler
    s0_lb = T.(problem.u0)
    s0_ub = s0_lb

    return DiffEqEnv(problem, reward_fn, n_actions, dt, s0_lb, s0_ub; T, kwargs...)
end

"""
RLBase Traits
"""
RLBase.NumAgentStyle(env::DiffEqEnv) = SINGLE_AGENT
RLBase.DynamicStyle(env::DiffEqEnv) = SEQUENTIAL # single agent
RLBase.ActionStyle(env::DiffEqEnv) = MINIMAL_ACTION_SET # all actions considered legal
RLBase.InformationStyle(env::DiffEqEnv) = PERFECT_INFORMATION # single agent
RLBase.StateStyle(env::DiffEqEnv) = Observation{Any}() # can be set to InternalState{Any}() in case of FullStateObservation()
RLBase.RewardStyle(env::DiffEqEnv) = STEP_REWARD # can in some cases be set to TERMINAL_REWARD
RLBase.UtilityStyle(env::DiffEqEnv) = GENERAL_SUM
RLBase.ChanceStyle(env::DiffEqEnv) = DETERMINISTIC # TODO: check if correct w/ random ICs

"""
RLBase interface for use with ReinforcementLearning.jl
"""
Random.seed!(env::DiffEqEnv, seed) = Random.seed!(env.rng, seed)

RLBase.action_space(env::DiffEqEnv) = env.action_space
RLBase.reward(env::DiffEqEnv) = env.reward
RLBase.is_terminated(env::DiffEqEnv) = env.done

# The following functions return observed state, not MDP state!
RLBase.state_space(env::DiffEqEnv) = env.observation_space
function RLBase.state(env::DiffEqEnv)
    return length(env.observation) > 1 ? env.observation : first(env.observation)
end

# Small patch to enable sampling on Intervals
# e.g. used to sample actions on action_space for random policy
function Random.rand(rng::AbstractRNG, s::Interval)
    return rand(rng) * (s.right - s.left) + s.left
end

function RLBase.reset!(env::DiffEqEnv{T}) where {T<:Real}
    # Reset environment
    env.state = T.(env.ic_sampler())
    env.observation = T.(env.observation_fn(env.state, NaN))
    env.reward = T(-Inf)
    env.done = false
    env.steps = 0
    env.t = env.ode_params.problem.tspan[1]
    return nothing
end

function (env::DiffEqEnv{T})(action) where {T<:Real}
    # type of action must fit with type of env.action_space
    action = T.(action)

    # unpack Array{T,1} with single value for use in ContinuousSpace{T}
    if length(action) == 1
        action = action[]
    end

    @assert action ∈ env.action_space

    # Remake ODEProblem over new tspan
    t_end = env.t + env.ode_params.dt
    tspan = (env.t, t_end)
    prob = remake(env.ode_params.problem; u0=env.state, tspan=tspan, p=action)

    # Integrate ODE
    if isadaptive(env.ode_params.solver)
        sol = solve(prob, env.ode_params.solver; env.ode_params.solve_args...)
    else # add timestep argument dt for fixed step-width solvers
        sol = solve(prob, env.ode_params.solver; env.ode_params.solve_args..., dt=t_end)
    end
    state_next = T.(sol.u[1]) # unpack Array{Array{T,1},1}

    # Update environment buffer
    env.observation = T.(env.observation_fn(state_next, action))
    env.reward = env.reward_fn(env.state, action, state_next) # update before env.state!
    env.state = state_next
    env.t = t_end # update before env.done!
    env.done = env.t >= env.ode_params.problem.tspan[2]
    env.steps += 1
    return nothing
end
