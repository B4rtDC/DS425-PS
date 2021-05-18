# abstract definition
abstract type AbstractMarkovDecisionProcess end

## some constants
# transition probs
const αts = 1/2
const αtw = 4/7
const αew = 4/5
const αewo = 1/2
# transition rewards
const rtse = -0.1
const rtst = -0.1
const rtwt = 1.0
const rtwoe = -0.5
const rtwoet = -0.3
const rewt = 1.4
const rewe = 1.6
const rewoh = -0.6
const rewoe = -0.2
const rhwt = 10

# our problem
struct HumanMDP <: AbstractMarkovDecisionProcess
    initial_state::String
    states::Set{String}
    rewards::Dict
    transitions::Dict
    γ::Float64
    function HumanMDP(initial_state::String, γ::Float64)
        # states (s)
        states = Set(["tired", "energized","healthier"])
        # transition
        trans = Dict(
            "tired" => Dict("sleep"=> [(αts,"energized"),(1-αts,"tired")],
                            "work" => [(1, "tired")],
                            "workout" => [(αtw,"energized"),(1-αtw,"tired")]),
            "energized" => Dict("work"=> [(αew,"tired"),(1-αew,"energized")],
                                "workout"=> [(αewo,"healthier"),(1-αewo,"energized")]),
            "healthier" => Dict("work"=> [(1, "tired")]))
        # rewards R(s,a,s')
        rewards = Dict(
            "tired" => Dict("sleep"=> [(rtse,"energized"),(rtst,"tired")],
                            "work" => [(rtwt, "tired")],
                            "workout" => [(rtwoe,"energized"),(rtwoet,"tired")]),
            "energized" => Dict("work"=> [(rewt,"tired"),(rewe,"energized")],
                                "workout"=> [(rewoh,"healthier"),(rewoe,"energized")]),
            "healthier" => Dict("work"=> [(rhwt, "tired")]))
        return new(initial_state, states, rewards, trans, γ)
    end
end

function actions(mdp::HumanMDP, state)
    return keys(mdp.transitions[state])
end

function transition_model(mdp::HumanMDP, state::String, action::String)
    return mdp.transitions[state][action]
end

function reward(mdp::HumanMDP, state::String, action::String, ns::String)
    idx = findfirst(x-> x[2]==ns, mdp.rewards[state][action])
    return mdp.rewards[state][action][idx][1]
end


"""
    value_iteration(mdp::HumanMDP; epsilon::Float64=0.001)

Return the utilities of the MDP's states as a Dict by applying the value iteration algorithm 
on the given Markov decision process 'mdp' and a arbitarily small positive number 'epsilon'.

Modified version that works with R(s,a,s')
"""
function value_iteration(mdp::HumanMDP; ϵ::Float64=0.001)
    # initialise utility to zero for all states
    U_prime = Dict(Pair(state, 0.0) for state in mdp.states)
    while true
        U = copy(U_prime)
        δ = 0.0
        # go over all possible state
        for state in mdp.states
            # possible outcomes: Array[ Pair(a => [(T(s,a,s', s'), (T(s,a,s', s')]) ]
            outcomes = [Pair(action,transition_model(mdp, state, action)) for action in actions(mdp, state)]
            expectations = map(x-> sum( T*(reward(mdp, state, x.first, ns) + mdp.γ * U[ns]) for (T,ns) in x.second), outcomes)
            U_prime[state] = maximum(expectations)
            δ = max(δ, abs(U_prime[state] - U[state]))
        end
        if (δ < ((ϵ * (1 - mdp.γ))/mdp.γ))
            return U_prime
        end
    end
end

"""
    policy_extraction(mdp::HumanMDP; epsilon::Float64=0.001)

Return the optimal policy given the computed utilities

Modified version that works with R(s,a,s')
"""
function policy_extraction(mdp::HumanMDP, U::Dict)
	pi = Dict()
	for state in mdp.states
        # possible outcomes: Array[ Pair(a => [(T(s,a,s', s'), (T(s,a,s', s')]) ]
        outcomes = [Pair(action,transition_model(mdp, state, action)) for action in actions(mdp, state)]
        expectations = map(x-> sum( T*(reward(mdp, state, x.first, ns) + mdp.γ * U[ns]) for (T,ns) in x.second), outcomes)
		pi[state] = outcomes[argmax(expectations)].first
	end
	return pi
end


h = HumanMDP("tired", 0.)
s = "tired"
a = "workout"
ns = "tired"
actions(h,s)
transition_model(h, s, a)
reward(h, s, a, ns)


# γ = 0 => only look at immediate (local) gains
@info "running the model using γ = 0"
h = HumanMDP("tired", 0.)
V_star = value_iteration(h)
π_star = policy_extraction(h,V_star)
display(π_star)
# γ > 0 => also consider neighbors
γ = 0.85
@info "running the model using γ = $(γ)"
H = HumanMDP("tired", γ)
VV_star = value_iteration(H)
ππ_star = policy_extraction(H,VV_star)
display(ππ_star)