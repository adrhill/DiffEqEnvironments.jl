"""
Struct for deterministic state feedback policies of type ``a=\\pi(s)``
"""
struct FeedbackPolicy <: AbstractPolicy
    π 
end

(p::FeedbackPolicy)(env) = p.π(get_state(env))

"""
Linear state feedback policy ``a=\\pi(s)=Ks``
"""
LinearFeedbackPolicy(K) = FeedbackPolicy(s -> K * s)
LinearFeedbackPolicy(K, a_lb, a_ub) = FeedbackPolicy(s -> clamp.(K * s, a_lb, a_ub))    

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
    # Discretize continuous state-space model 
    Ā = exp(A * dt)
    B̄ = A \ (Ā - I) * B
    
    # Compute discrete LQR gain matrix
    K = dlqr(Ā, B̄, Q, R)
end