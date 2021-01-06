"""
Struct for deterministic state feedback policies of type ``a=\\pi(s)``
"""
struct FeedbackPolicy <: AbstractPolicy
    π
end

"""
Implement interface required for RLCore agent
"""
(p::FeedbackPolicy)(env) = p.π(state(env))

# policy stays constant
RLBase.update!(p::FeedbackPolicy, experience) = nothing
RLBase.update!(
    p::FeedbackPolicy,
    traj::AbstractTrajectory,
    env::AbstractEnv,
    stage::AbstractStage,
) = nothing
RLBase.update!(
    p::FeedbackPolicy,
    traj::AbstractTrajectory,
    env::AbstractEnv,
    stage::PreActStage,
) = nothing

"""
Linear state feedback policy ``a=\\pi(s)=Ks``
"""
LinearFeedbackPolicy(K) = FeedbackPolicy(s -> K * reshape(s, :))
LinearFeedbackPolicy(K, a_lb, a_ub) = FeedbackPolicy(s -> clamp.(K * reshape(s, :), a_lb, a_ub))

"""
Discrete LQR state feedback policy.

Discretizes the continuous sytem ``\\dot{s}=As+Ba`` and
calculates the optimal state-feedback law ``a_k = -K s_k`` that
minimizes the cost function:
``J = \\sum_{k=0}^\\infty(s_k'Qs_k + a_k'Ra_k)``
For the discretized time model ``s_{k+1} = \\bar{A}s_k + \\bar{B}a_k``.
"""
function LQRPolicy(A, B, Q, R, dt)
    K = _dlqr_from_cont_ss(A, B, Q, R, dt)
    return LinearFeedbackPolicy(-K)
end

function LQRPolicy(A, B, Q, R, dt, a_lb, a_ub)
    K = _dlqr_from_cont_ss(A, B, Q, R, dt)
    return LinearFeedbackPolicy(-K, a_lb, a_ub)
end

function _dlqr_from_cont_ss(A, B, Q, R, dt)
    # Build discrete state-space model using place-holder matrices C, D
    nx = size(A)[1]
    na = size(B[:,:])[2]

    C = zeros(1, nx)
    D = zeros(1, na)
    sysd = ss(A, B, C, D, dt)

    # Compute discrete LQR gain matrix
    K = lqr(sysd, Q, R)
end
