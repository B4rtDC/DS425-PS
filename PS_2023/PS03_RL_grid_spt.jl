# obtain the gridworld MDP methods
include("./PS02_MDP_spt.jl")

# ----------------------------------------------------------- #
#       Reinforcement Learning - Generic Elements             #
# ----------------------------------------------------------- #

abstract type ReinformentLearningMethod end
	
"""
    take_single_action(mdp::T, s, a) where T<:AbstractMarkovDecisionProcess

Given the MDP, a state `s` and a desired action `a`, obtain the new state `s'`. This happens at random, using the model's transition probabilities.
"""
function take_single_action(mdp::T, s, a) where T<:AbstractMarkovDecisionProcess
    x = rand() # random number (reference value)
    p_cum = 0. # cumulative probability
    for (p, newstate) in transition_model(mdp, s, a)
        p_cum += p
        if x < p_cum
            return newstate
        end
    end
end
	

"""
    single_trial!(rlm::V, mdp::T) where {V<:ReinformentLearningMethod, T<:AbstractMarkovDecisionProcess}

Run a single trial until a terminal state is reached. The `ReinformentLearningMethod` is updated during the process.

Return the sequence of states visited during the trial.
"""
function single_trial!(rlm::V, mdp::T) where {V<:ReinformentLearningMethod, T<:AbstractMarkovDecisionProcess}
    s = mdp.initial
    trial = [s] # not required, for didactic purpose only
    # reset method for new trial (if required, method should be implemented)
    init!(rlm)
    while true
        # get reward from current state
        r = reward(mdp, s) 
        # transfer state and reward to ReinformentLearningMethod and obtain the action from the policy
        a = learn!(rlm, s, r)
        if isnothing(a)
            break
        end
        # update the state
        s = take_single_action(mdp, s, a)
        push!(trial, s) # not required, for didactic purpose only
    end
    # update the utilities (if required, method should be implemented)
    update!(rlm)
    
    return trial
end

# ----------------------------------------------------------- #
#  Passive reinforcement Learning - Direct Utility Estimation #
# ----------------------------------------------------------- #
"""
    DirectUtilityEstimation

Reinforcement learning method making use of direct utility estimation
"""
mutable struct DirectUtilityEstimation <: ReinformentLearningMethod
    "policy used for RL"
    policy::Dict
    "estimate for utilities (running average over trials)"
    U::Dict 
    "buffer for single trial"
    buffer::Dict
    "number of observations for each state"
    n::Dict
end

function DirectUtilityEstimation(Π) 
    DirectUtilityEstimation(Π, 	Dict{eltype(keys(Π)), Float64}(),
                                Dict{eltype(keys(Π)), Vector{Float64}}(),
                                Dict{eltype(keys(Π)), Int64}())
end

"""
    init!(rlm::DirectUtilityEstimation)

reset the buffer for direct utility estimation
"""
function init!(rlm::DirectUtilityEstimation)
    rlm.buffer = Dict{eltype(keys(rlm.policy)), Vector{Float64}}()
end

"""
    update!(rlm::DirectUtilityEstimation)

after a single trial, update the utility estimates by maintaining a running average
"""
function update!(rlm::DirectUtilityEstimation)
    for state in keys(rlm.buffer)
        k = length(get!(rlm.buffer,state,[]))
        rlm.U[state] = (get!(rlm.U, state, 0.) * get!(rlm.n, state, 0) + sum(rlm.buffer[state]))/ (get!(rlm.n, state, 0) + k)
        rlm.n[state] += k
    end
end


"""
    learn!(rlm::DirectUtilityEstimation, s, r) 

Update the utilies by using the state `s` and the associated reward `r`
"""
function learn!(rlm::DirectUtilityEstimation, s, r; γ::Float64=1.) 
    # update all known estimates
    for state in keys(rlm.buffer)
        #rlm.buffer[state] .*= γ # discount
        rlm.buffer[state] .+= r # update
    end
    # insert new state if not seen yet
    vals = get!(rlm.buffer, s, Float64[])
    push!(vals, r)
    
    # return action from the policy
    return rlm.policy[s] 
end

# ----------------------------------------------------------------- #
#  Passive reinforcement Learning - Sample Based Utility Estimation #
# ----------------------------------------------------------------- #

mutable struct SampleBasedEstimation{T} <:ReinformentLearningMethod where T <: AbstractMarkovDecisionProcess
    "current state"
    state
    "current action"
    action
    "policy used for RL"
    policy::Dict
    "estimate for utilities"
    U::Dict 
    "internal represenation of the MDP we are learning"
    mdp::T
    "counts of seeing action a in state s"
    N_sa::Dict
    "counts of action a and state s leading to s prime"
    N_s_prime_sa::Dict
    function SampleBasedEstimation(Π::Dict, mdp::U) where {U <: AbstractMarkovDecisionProcess}
        internal_mdp = MarkovDecisionProcess(mdp.initial, mdp.actions, mdp.terminal_states, Dict(), gamma=mdp.gamma)
        return new{typeof(internal_mdp)}(nothing, nothing, Π, Dict(), internal_mdp, Dict(), Dict())
    end
end

#function SampleBasedEstimation(Π::Dict, mdp::T) where {T <: AbstractMarkovDecisionProcess}
#    return SampleBasedEstimation(nothing, nothing, Π, Dict(), mdp, Dict(), Dict())
#end

"""
    transition_model(rlm::SampleBasedEstimation, state, action)

Return a list of (P(s'|s, a), s') pairs given the state 's' and action 'a'.
"""
function transition_model(rlm::SampleBasedEstimation, state, action)
    return collect((v, k) for (k, v) in get!(rlm.mdp.transitions, (state, action), Dict()))
end


"""
    policy_evaluation!(rlm::SampleBasedEstimation; k::Int64=20)

Return the updated utilities of the MDP's states by applying the modified policy iteration
algorithm on the given Markov decision process 'mdp', utility function 'U', policy 'pi',
and number of Bellman updates to use 'k'.
"""
function policy_evaluation!(rlm::SampleBasedEstimation; k::Int64=20)
    for i in 1:k
        for state in rlm.mdp.states
            if length(transition_model(rlm, state, rlm.policy[state])) != 0
                rlm.U[state] = rlm.mdp.reward[state] + rlm.mdp.gamma * sum(p * rlm.U[state_prime] for (p, state_prime) in transition_model(rlm, state, rlm.policy[state]))
            else
                rlm.U[state] = rlm.mdp.reward[state] 
            end
        end
    end
    return rlm.U
end

"""
    learn!(rlm::SampleBasedEstimation, s, r) 

Update the utilies by using the state `s` and the associated reward `r`
"""
function learn!(rlm::SampleBasedEstimation, s, r)
    push!(rlm.mdp.states, s)

    # check if s is a new state
    if !haskey(rlm.mdp.reward, s)
        rlm.U[s] = r
        rlm.mdp.reward[s] = r
    end
    
    # update our estimates of the transition probabilities
    if !isnothing(rlm.state)
        # update the counts for non-terminal states
        rlm.N_sa[(rlm.state, rlm.action)] = get!(rlm.N_sa, (rlm.state, rlm.action), 0) + 1
        rlm.N_s_prime_sa[(s, rlm.state, rlm.action)] = get!(rlm.N_s_prime_sa, (s, rlm.state, rlm.action), 0) + 1
        # update the transition model given the new observations
        for t in [res for ((res, state, action), counts) in rlm.N_s_prime_sa if ((state, action) == (rlm.state, rlm.action) && counts ≠ 0)]
            get!(rlm.mdp.transitions, (rlm.state, rlm.action), Dict())[t] = rlm.N_s_prime_sa[(t, rlm.state, rlm.action)] / rlm.N_sa[(rlm.state, rlm.action)]
        end
    end

    # update the policy (with policy iteration)
    policy_evaluation!(rlm)
    
    # update state
    if s ∈ rlm.mdp.terminal_states
        rlm.state = nothing
        rlm.action = nothing
    else
        rlm.state = s
        rlm.action = rlm.policy[s]
    end
    
end

init!(rlm::SampleBasedEstimation) = nothing
update!(rlm::SampleBasedEstimation) = nothing



# ----------------------------------------------------------------- #
#             Active reinforcement Learning - Q-learning            #
# ----------------------------------------------------------------- #
mutable struct Qlearner <:ReinformentLearningMethod
    "current state"
    state
    "current action"
    action
    "current reward"
    reward
    "possible actions"
    actions::Set
    "terminal states"
    terminal_states::Set
    "optimistic estimate best possible reward"
    R⁺::Float64
    "Q values"
    Q::Dict 
    "γ value"
    γ::Float64
    "exploration function"
    f::Function
    "learning rate α"
    α::Function
    "counts of seeing action a in state s"
    N_sa::Dict
    "try (state, action) at least N_e times"
    N_e::Int64
end

function Qlearner(mdp::T; N_e::Int=10, R⁺::Float64=1., 
                    α::Union{Nothing, Function}=nothing, exploration_function::Function=simple_exploration_function) where T<: AbstractMarkovDecisionProcess
    
    α_fun = isnothing(α) ? n->1/(n+1) : α
    return Qlearner(nothing, 
                    nothing, 
                    nothing,
                    mdp.actions, 
                    mdp.terminal_states, 
                    R⁺, 
                    Dict(), 
                    mdp.gamma, 
                    exploration_function, 
                    α_fun, 
                    Dict(), 
                    N_e)
end

function simple_exploration_function(rln::Qlearner, u, n)
    return n < rln.N_e ? rln.R⁺ : u
end

other_α_fun(n) = 60 / (59 + n)

actions(rlm::Qlearner, state) = state ∈ rlm.terminal_states ? Set([nothing]) : rlm.actions

"""
    learn!(rlm::Qlearner, sᶥ, rᶥ) 

Update the Q-values by using the state `s` and the associated reward `r`
"""
function learn!(rlm::Qlearner, sᶥ, rᶥ) 
    s, a, r, Q, α, N_sa, γ, f = rlm.state, rlm.action, rlm.reward, rlm.Q, rlm.α, rlm.N_sa, rlm.γ, rlm.f

    # update and evaluate Qvalues
    if sᶥ ∈ rlm.terminal_states
        #@warn "terminal state reached: ($(sᶥ), $(rᶥ))"
        Q[sᶥ, nothing] =  rᶥ
    end
    
    if !isnothing(s)
        # increase counts (used for exploration function)
        N_sa[(s, a)] = get!(N_sa, (s, a), 0) + 1
        # compute: Q(s,a) = Q(s,a) + α(n_sa)[r +γ*max(q(s',a')) - Q(s,a)]
        get!(Q, (s,a), 0.)
        maxval = reduce(max, [get!(Q, (sᶥ,action), 0.) for action in actions(rlm, sᶥ)])
        Q[(s,a)] += α(N_sa[(s,a)]) * (rᶥ + γ * maxval  - Q[(s,a)])
    end
    
    # update state 
    if !isnothing(sᶥ) && (sᶥ ∈ rlm.terminal_states)
        rlm.state = nothing
        rlm.action = nothing
        rlm.reward = nothing
    else
        rlm.state = sᶥ
        rlm.reward = rᶥ
        # next action accounts for exploration function
        rlm.action = argmax(aᶥ -> f(rlm, get!(Q, (sᶥ,aᶥ), 0.), get!(N_sa, (sᶥ,aᶥ), 0)), actions(rlm, sᶥ) )
    end

    return rlm.action
end

init!(rlm::Qlearner) = nothing
update!(rlm::Qlearner) = nothing
