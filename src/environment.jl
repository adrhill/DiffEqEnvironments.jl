mutable struct DiffEqEnv{T,F} <: AbstractEnv
    n_states::Int                           # number of states
    n_observations::Int                     # number of observed states
    n_actions::Int                          # number of inputs / dimensionality of action
    #== Parameters for DifferentialEquations.jl ==#
    ode::F                                  # differential equation to simulate
    s0::Union{T, Vector{T}}                 # initial condition
    dt::T                                   # time step size
    t_end::T                                # time until which agent is run
    solver::AbstractODEAlgorithm            # solver to use to solve ODE
    reltol::T                               # relative tolerance for ODE solver
    abstol::T                               # absolute tolerance for ODE solver
    #== Parameters for ReinforcementLearning.jl ==#
    observation_space::MultiContinuousSpace{Vector{T}}
    action_space::MultiContinuousSpace{Vector{T}}
    o0::Union{T, Vector{T}}                 # initial observation
    observation_fn::ObservationFunction     # obvervation o=f(s)
    reward_fn::RewardFunction               # reward function r(s,a,s')
    #== Buffer for previous step ==#
    state::Union{T, Vector{T}}              # current state
    observation::Union{T, Vector{T}}        # current observation
    action::Union{Nothing, T, Vector{T}}    # last action
    reward::Union{Nothing, T}               # last scalar reward
    done::Bool                              # true if in terminal state 
    steps::Int                              # counter for steps taken in episode
end

function DiffEqEnv(ode, 
                   reward_fn::RewardFunction,
                   n_actions::Int,
                   s0::Union{Real, Vector{Real}}, 
                   dt::Real,
                   t_end::Real;
                   #== Keyword arguments ==#
                   observation_fn::ObservationFunction=FullObservationFunction(),
                   o_ub::Union{Nothing, Vector{<:Real}}=nothing, # upper bound for observation space
                   o_lb::Union{Nothing, Vector{<:Real}}=nothing, # lower bound for observation space
                   a_ub::Union{Nothing, Vector{<:Real}}=nothing, # upper bound for action space
                   a_lb::Union{Nothing, Vector{<:Real}}=nothing, # lower bound for action space
                   solver::DiffEqBase.AbstractODEAlgorithm=Tsit5(), 
                   reltol::Real=1e-8, 
                   abstol::Real=1e-8,
                   T = Float32
                   )
    #== Constructor for the mutable struct DiffEqEnv.
    This is the interface for the user ==#
    
    # Turn scalar state into 1D-vector
    if s0 isa Real 
        s0=[s0]
    end
    
    o0 = observation_fn.f(s0) #initial observations
    n_states = length(s0)
    n_observations = length(o0)

    # Set default upper and lower bounds for MultiContinuousSpace
    maxval = typemax(T)
    if isnothing(o_lb)
        o_lb= -maxval * ones(T, n_states)
    end
    if isnothing(o_ub)
        o_ub= maxval * ones(T, n_states)
    end
    if isnothing(a_lb)
        a_lb= -maxval * ones(T, n_actions)
    end
    if isnothing(a_ub)
        a_ub= maxval * ones(T, n_actions)
    end

    # Check inputs for dimension mismatches
    size(o_lb) == size(o_ub) ||
        throw(ArgumentError("$(size(o_lb)) != $(size(o_ub)), size must match"))

    size(o_lb) == size(o0) ||
        throw(ArgumentError("$(size(o_lb)) != $(size(o0)), size must match"))

    size(a_lb) == size(a_ub) ||
        throw(ArgumentError("$(size(a_lb)) != $(size(a_ub)), size must match"))

    size(a_lb) == n_actions ||
        throw(ArgumentError("$(size(a_lb)) != $(n_actions), size must match"))

    # Set observation and action spaces
    observation_space = MultiContinuousSpace(o_lb, o_ub) #TODO: imple
    action_space = MultiContinuousSpace(a_lb, a_ub)

    # Initialize previous step buffer
    state = s0
    observation = o0
    action = nothing
    reward = nothing
    done = false
    steps = 0

    env = DiffEqEnv{T}(
            n_states, n_observations, n_actions,
            #== Parameters for DifferentialEquations.jl ==#
            ode, s0, dt, t_end,
            solver, reltol, abstol, p,
            #== Parameters for ReinforcementLearning.jl ==#
            observation_space, action_space,
            observation_fn, reward_fn,
            #== Buffer for previous step ==#
            state, observation, action, reward, done,
            steps)
    return env
end

RLBase.get_actions(env::DiffEqEnv) = env.action
RLBase.get_state(env::DiffEqEnv) = env.observation # attention: returns _observed_ state
RLBase.get_reward(env::DiffEqEnv) = env.reward
RLBase.get_terminal(env::DiffEqEnv) = env.done

function reset!(env::DiffEqEnv)
    # Reset environment
    env.steps_taken = 0
    env.state = env.s0
    env.observation = env.o0
    env.reward = nothing
    env.done = false
end

function (env::DiffEqEnv)(action::Union{Real, Vector{Real}})
    @assert action in env.action_space

    # Integrate ODE
    tspan = (0, env.dt)
    prob = ODEProblem(env.ode!, env.s0, tspan, p=action)
    sol = solve(prob, env.solver,
            reltol=env.reltol, abstol=env.abstol,
            save_everystep=false,save_start=false)    
    state_next = sol.u 
    
    # Update environment state, termination status & reward
    env.steps += 1
    env.state = state_next
    env.observation = env.observation_fn.f(state_next)
    env.action = action
    env.reward = env.reward_fn.f(env.state, action, state_next)
    env.done = (env.steps * env.dt) >= env.t_end

    return nothing
 end