"""
Base struct for all DiffEqEnvironments.jl rewards.
"""
struct RewardFunction
    f
end

(rf::RewardFunction)(s,a,s_next) = rf.f(s, a, s_next)

"""  
Default reward of form ``r(s,a,s')``.
"""
SASReward(r) = RewardFunction(r)

"""
Reward of type ``r(s,a)``.
"""
SAReward(r) = RewardFunction((s, a, _) -> r(s, a))

"""
Reward of type ``r(a,s')``.
"""
ASReward(r) = RewardFunction((_, a, s) -> r(a, s))    

"""
Quadratic reward of type ``r(s,a)=-s^T\\mathbf{Q}s - s^T\\mathbf{Q}s``, 
commonly used in optimal control / LQR.
"""
function QuadraticReward(Q, R)
    r(_, a, s) = -s' * Q * s - a' * R * a
    return RewardFunction(r)
end

"""
Reward ``r(s')=0``` if ``s'`` is terminal, ``r(s')=-1`` else
"""
function DecrementingReward(s_next, condition_fn)
    r(s, a, s_next) = condition_fn(s_next) ? 0 : -1
    return RewardFunction(r)
end