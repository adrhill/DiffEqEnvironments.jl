""" 
Struct holding parameters for ODE solver
"""
mutable struct DiffEqParams{T}
    problem::ODEProblem             # ODEProblem which is simulated
    dt::T                           # time step size
    solver::AbstractODEAlgorithm    # solver to use to solve ODE
    solve_args::Dict                # dictionary holding kwargs for ODE solve command
end

mutable struct DiffEqEnv{T} <: AbstractEnv
    ode_params::DiffEqParams{T}
    ic_sampler::ICSampler
    # Parameters for ReinforcementLearning
    observation_space::Union{ContinuousSpace{T},MultiContinuousSpace{Vector{T}}}
    action_space::Union{ContinuousSpace{T},MultiContinuousSpace{Vector{T}}}
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
    T=Float64
    )
    # Save solver arguments
    solve_args = Dict{Symbol,Any}(
        :reltol => reltol, 
        :abstol => abstol,
        :save_everystep => false, 
        :save_start => false) # only output terminal value

    ode_params = DiffEqParams{T}(problem, dt, solver, solve_args)
    
    # Set inital state and observation
    ic_sampler = UniformSampler(s0_lb, s0_ub)
    s0 = T.(ic_sampler()) # Convert type
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
                lb = - ub
            end
        end

        # Check if sizes match
        length(lb) == length(ub) ||
            throw(ArgumentError("$(size(lb)) != $(size(ub)), size must match"))
        length(lb) == dim ||
            throw(ArgumentError("$(size(lb)) != $(dim), size must match"))
        
        # Return ContinuousSpace or MultiContinuousSpace
        if lb isa Real
            return ContinuousSpace(T(lb), T(ub))
        elseif lb isa Vector{<:Real}
            return MultiContinuousSpace(T.(lb), T.(ub))
        end
    end
    
    # Set observation and action spaces
    observation_space = set_space(o_lb, o_ub, n_observations)
    action_space = set_space(a_lb, a_ub, n_actions)

    # Initialize buffer holding previous step
    state = s0
    observation = o0
    action = nothing
    reward = nothing
    done = false
    steps = 0
    t = problem.tspan[1]

    env = DiffEqEnv{T}(ode_params, ic_sampler,
            observation_space, action_space,
            observation_fn, reward_fn,
            state, observation, action, reward, done, steps, t)
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
    #= Keyword arguments =#
    observation_fn::ObservationFunction=FullStateObservation(),
    o_ub::Union{Nothing,Real,Vector{<:Real}}=nothing, # upper bound for observation space
    o_lb::Union{Nothing,Real,Vector{<:Real}}=nothing, # lower bound for observation space
    a_ub::Union{Nothing,Real,Vector{<:Real}}=nothing, # upper bound for action space
    a_lb::Union{Nothing,Real,Vector{<:Real}}=nothing, # lower bound for action space
    solver::DiffEqBase.AbstractODEAlgorithm=Euler(), 
    reltol::Real=1e-8, 
    abstol::Real=1e-8, # TODO: add more kwargs for integrator
    T=Float64
    )

    #= Set bounds for ICs to same values
    This ensures that the same IC gets sampled by env.ic_sampler =#
    s0_lb = T.(problem.u0)
    s0_ub = s0_lb
    
    return DiffEqEnv(problem, reward_fn, n_actions, dt, s0_lb, s0_ub;
        observation_fn=observation_fn,
        o_ub=o_ub, o_lb=o_lb, a_ub=a_ub, a_lb=a_lb,
        solver=solver, reltol=reltol, abstol=abstol, T=T)

end

"""
RLBase interface for use with ReinforcementLearning.jl
"""
RLBase.get_actions(env::DiffEqEnv) = env.action_space
RLBase.get_state(env::DiffEqEnv) = env.observation # returns observed state, not markov state!
RLBase.get_reward(env::DiffEqEnv) = env.reward
RLBase.get_terminal(env::DiffEqEnv) = env.done

function RLBase.reset!(env::DiffEqEnv{T}) where {T <: Real}
    # Reset environment
    env.state = T.(env.ic_sampler())
    env.action = nothing
    env.observation = T.(env.observation_fn(env.state, env.action))
    env.reward = nothing
    env.done = false
    env.steps = 0
    env.t = env.ode_params.problem.tspan[1]

    return nothing
end

function (env::DiffEqEnv{T})(action) where {T <: Real}
    # type of action must fit with type of env.action_space
    action = T.(action)

    # unpack Array{T,1} with single value for use in ContinuousSpace{T}
    if length(action) == 1 
        action = action[] 
    end
    
    @assert action in env.action_space

    # Remake ODEProblem over new tspan
    t_end =  env.t + env.ode_params.dt
    tspan = (env.t, t_end)
    prob = remake(env.ode_params.problem; 
                u0=env.state, tspan=tspan, p=action)

    # Integrate ODE
    if isadaptive(env.ode_params.solver)
        sol = solve(prob, env.ode_params.solver; env.ode_params.solve_args...)
    else # add timestep argument dt for fixed step-width solvers
        sol = solve(prob, env.ode_params.solver; env.ode_params.solve_args..., dt=t_end)
    end
    state_next = T.(sol.u[1]) # unpack Array{Array{T,1},1}
    
    # Update environment buffer
    env.observation = T.(env.observation_fn(state_next, action))
    env.action = T.(action)
    env.reward = env.reward_fn(env.state, action, state_next) # update before env.state!
    env.state = state_next 
    env.t = t_end # update before env.done!
    env.done = env.t >= env.ode_params.problem.tspan[2]
    env.steps += 1

    return nothing
 end