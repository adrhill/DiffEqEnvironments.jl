""" 
Struct holding parameters for ODE solver
"""
mutable struct DiffEqParams{T}
    problem::ODEProblem             # ODEProblem which is simulated
    dt::T                           # time step size
    solver::AbstractODEAlgorithm    # solver to use to solve ODE
    reltol::T                       # relative tolerance for ODE solver
    abstol::T                       # absolute tolerance for ODE solver
end

mutable struct DiffEqEnv{T} <: AbstractEnv
    ode_params::DiffEqParams{T}
    # Parameters for ReinforcementLearning
    observation_space::MultiContinuousSpace{Vector{T}}
    action_space::MultiContinuousSpace{Vector{T}}
    # Observation & reward functions
    observation_fn::ObservationFunction     # obvervation o=f(s)
    reward_fn::RewardFunction               # reward function r(s,a,s')
    # Buffer for previous transition
    state::Union{T,Vector{T}}               # current state
    observation::Union{T,Vector{T}}         # current observation
    action::Union{Nothing,T,Vector{T}}      # last action
    reward::Union{Nothing,T}                # last scalar reward
    done::Bool                              # true if in terminal state 
    steps::Int                              # counter for steps taken in episode
    t::T                                    # current time
end

"""
User-facing constructor for the DiffEqEnv
"""
function DiffEqEnv(problem::ODEProblem, 
            reward_fn::RewardFunction,
            n_actions::Int,
            dt::Real;
            #= = Keyword arguments = =#
            observation_fn::ObservationFunction=FullObservationFunction(),
            o_ub::Union{Nothing,Vector{<:Real}}=nothing, # upper bound for observation space
            o_lb::Union{Nothing,Vector{<:Real}}=nothing, # lower bound for observation space
            a_ub::Union{Nothing,Vector{<:Real}}=nothing, # upper bound for action space
            a_lb::Union{Nothing,Vector{<:Real}}=nothing, # lower bound for action space
            solver::DiffEqBase.AbstractODEAlgorithm=Tsit5(), 
            reltol::Real=1e-8, 
            abstol::Real=1e-8,
            # TODO: add more kwargs for integrator
            T=Float32
            )
    
    # Turn scalar state into 1D-vector
    s0 = problem.u0
    if s0 isa Real 
        s0 = [s0]
    end
    s0 = T.(s0) # convert type
    
    o0 = observation_fn.f(s0) # initial observations
    n_states = length(s0)
    n_observations = length(o0)

    # Set default upper and lower bounds for MultiContinuousSpace
    maxval = typemax(T)
    if isnothing(o_lb)
        o_lb = -maxval * ones(T, n_states)
    end
    if isnothing(o_ub)
        o_ub = maxval * ones(T, n_states)
    end
    if isnothing(a_lb)
        a_lb = -maxval * ones(T, n_actions)
    end
    if isnothing(a_ub)
        a_ub = maxval * ones(T, n_actions)
    end

    # Check inputs for dimension mismatches
    size(o_lb) == size(o_ub) ||
        throw(ArgumentError("$(size(o_lb)) != $(size(o_ub)), size must match"))

    size(o_lb) == size(o0) ||
        throw(ArgumentError("$(size(o_lb)) != $(size(o0)), size must match"))

    size(a_lb) == size(a_ub) ||
        throw(ArgumentError("$(size(a_lb)) != $(size(a_ub)), size must match"))

    length(a_lb) == n_actions ||
        throw(ArgumentError("$(size(a_lb)) != $(n_actions), size must match"))

    # Set observation and action spaces
    observation_space = MultiContinuousSpace(o_lb, o_ub) # TODO: imple
    action_space = MultiContinuousSpace(a_lb, a_ub)

    # Initialize previous step buffer
    state = s0
    observation = o0
    action = nothing
    reward = nothing
    done = false
    steps = 0
    t = problem.tspan[1]

    ode_params = DiffEqParams{T}(problem, dt, solver, reltol, abstol)

    env = DiffEqEnv{T}(ode_params, 
            observation_space, action_space,
            observation_fn, reward_fn,
            state, observation, action, reward, done, steps, t)
    return env
end

RLBase.get_actions(env::DiffEqEnv) = env.action
RLBase.get_state(env::DiffEqEnv) = env.observation # returns observed state, not markov state!
RLBase.get_reward(env::DiffEqEnv) = env.reward
RLBase.get_terminal(env::DiffEqEnv) = env.done

function RLBase.reset!(env::DiffEqEnv{T}) where {T<:Real}
    # Reset environment
    env.state = T.(env.ode_params.problem.u0)
    env.observation = env.observation_fn(env.state) 
    env.action = nothing
    env.reward = nothing
    env.done = false
    env.steps = 0
    env.t = env.ode_params.problem.tspan[1]

    return nothing
end

function (env::DiffEqEnv)(action)
    @assert action in env.action_space

    # Remake ODEProblem over new tspan
    t_end =  env.t + env.ode_params.dt
    tspan = (env.t, t_end)
    prob = remake(env.ode_params.problem; 
                u0=env.state, tspan=tspan, p=action)

    # Integrate ODE
    sol = solve(prob, env.ode_params.solver,
            reltol=env.ode_params.reltol, 
            abstol=env.ode_params.abstol,
            save_everystep=false, save_start=false) # only save values at tspan[2]
    state_next = sol.u[1] # unpack Array{Array{T,1},1}
    
    # Update environment buffer
    env.observation = env.observation_fn(state_next)
    env.action = action
    env.reward = env.reward_fn(env.state, action, state_next) # update before env.state!
    env.state = state_next 
    env.t = t_end # update before env.done!
    env.done = env.t >= env.ode_params.problem.tspan[2]
    env.steps += 1

    return nothing
 end