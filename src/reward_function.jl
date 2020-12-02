"""
Base struct for all DiffEqEnvironments.jl rewards.
"""
struct RewardFunction
    f
end

(rf::RewardFunction)(s,a,s_next) = rf.f(s,a,s_next)


"""  
Default reward of form ``r(s,a,s')``.
"""
function SASReward(r)
    return RewardFunction(r)
end

"""
Reward of type ``r(s,a)``.
"""
function SAReward(r)
    _r(s,a,s_next) = r(s,a)
    return RewardFunction(_r)
end

"""
Reward of type ``r(a,s')``.
"""
function ASReward(r)
    _r(s,a,s_next) = r(a,s_next)
    return RewardFunction(_r)
end

"""
Quadratic reward of type ``r(s,a)=-s^T\\mathbf{Q}s - s^T\\mathbf{Q}s``, 
commonly used in optimal control / LQR.
"""
function QuadraticReward(Q,R)
    _r(s,a,s_next) = -s_next'*Q*s_next - a'*R*a
    return RewardFunction(_r)
end

"""
Reward ``r(s')=0``` if ``s'`` is terminal, ``r(s')=-1`` else
"""
function DecrementingReward(s_next, condition_fn)
    _r(s,a,s_next) = condition_fn(s_next) ? 0 : -1
    return RewardFunction(_r)
end