struct FeedbackPolicy <: AbstractPolicy
    π 
end

(p::FeedbackPolicy)(env) = p.π(get_state(env))

"""
Linear state feedback policy ``a=\pi(s)=Ks
"""
LinearFeedbackPolicy(K) = FeedbackPolicy(s-> K*s)
LinearFeedbackPolicy(K, a_lb, a_ub) = FeedbackPolicy(s-> clamp.(K * s, a_lb, a_ub))    

"""
LQR state feedback policy.

Calculates the optimal state-feedback law `as = Ks` that minimizes the cost function:
``J = \\int_0^\\infty s'Qs + a'Ra \\,dt``
For the continuous time model ``\\dot{s} = As + Ba``.
"""
LQRPolicy(A,B,Q,R) = LinearFeedbackPolicy(-lqr(A, B, Q, R))

function LQRPolicy(A,B,Q,R, a_lb, a_ub)
    K = dlqr(A, B, Q, R)
    return LinearFeedbackPolicy(K, a_lb, a_ub)
end

"""
Discrete LQR state feedback policy.

Calculates the optimal state-feedback law `a_k = K s_k` that
minimizes the cost function:
``J = \\sum_{k=0}^\\infty(s_k'Qs_k + a_k'Ra_k)``
For the discrte time model ``s_{k+1} = As_k + Ba_k``.
"""
DLQRPolicy(A,B,Q,R) = LinearFeedbackPolicy(-lqr(A, B, Q, R))

function DLQRPolicy(A,B,Q,R, a_lb, a_ub)
    K = dlqr(A, B, Q, R)
    return LinearFeedbackPolicy(K, a_lb, a_ub)
end