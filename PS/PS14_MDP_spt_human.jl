## --------------------------------------------- ##
## GENERIC PART
## --------------------------------------------- ##
abstract type AbstractMarkovDecisionProcess end

## --------------------------------------------- ##
## Code block human cycle
## --------------------------------------------- ##

## parameters for the model
# transition probabilities
const α_st = 0.9
const α_gt = 0.9
const α_we = 0.4
# rewards
const r_wt = 1
const r_gt = -0.5
const r_we = -0.2
const r_ge = 3.
const r_wh = 1
const r_st = -0.1

"""
    humancycle <: AbstractMarkovDecisionProcess

Struct having the same fields of the AbstractMarkovDecisionProcess.

Note that there is no terminal state.
"""
struct Humancycle <: AbstractMarkovDecisionProcess
    initial::String
    states::Set{String}
    transitions::Dict
    reward::Dict
    gamma::Float64
    function Humancycle(initial::String="energetic"; γ::Float64=0.0)
        # Check discount validity
        (0 < γ <= 1) ? nothing : error("MarkovDecisionProcess():\nThe gamma variable of an MDP must be between 0 and 1, the constructor was given ", γ, "!")

        # Set rewards (s, a, s') => reward
        reward = Dict()
        reward["tired"] =     Dict( ("work","tired") => r_wt, 
                                    ("workout","tired") => r_gt,
                                    ("workout","energetic") => r_gt,
                                    ("sleep","tired") => r_st,
                                    ("sleep","energetic") => r_st)
        reward["energetic"] = Dict( ("work", "tired") => r_we,
                                    ("work","energetic") => r_we,
                                    ("workout","healthier") => r_ge)
        reward["healthier"] = Dict( ("work","tired") => r_wh)

        # Set transition_model dict(state => dict(action => results))
        transitions = Dict()
        transitions["tired"] =      Dict("sleep" => [(α_st, "energetic"), (1-α_st, "tired")],
                                         "work" => [(1, "tired")],
                                         "workout" => [(α_gt, "energetic"), (1-α_gt, "energetic")]
                                         )
        transitions["energetic"] =  Dict("work" => [(α_we, "tired"),(1-α_we, "energetic")],
                                         "workout" => [(1, "healthier")])
        transitions["healthier"] =  Dict("work" => [(1, "tired")])

        # set possible states
        states = Set{String}(keys(transitions))

        return new(initial, states, transitions, reward, γ)
    end
end

"""
    actions(mdp::Humancycle, state::String)

returns possible action for out humancycle
"""
function actions(mdp::Humancycle, state::String)
    return keys(mdp.transitions[state])
end

"""
    reward(mdp::Humancycle, state, action, nextstate)

The reward does not only depend on the state, but also on the action and the next state: R(s,a,s')
"""
function reward(mdp::Humancycle, state::String, action::String, ns::String)
    return mdp.reward[state][(action,ns)]
end

"""
    transition_model(mdp::Humancycle, state, action, nextstate)

returns transtion results and probabilities: [(p, s')]
"""
function transition_model(mdp::Humancycle, state::String, action::String)
    return mdp.transitions[state][action]
end

"""
    value_iteration(mdp::Humancycle; epsilon::Float64=0.001)

Return the utilities of the MDP's states as a Dict by applying the value iteration algorithm (Fig. 17.4)
on the given Markov decision process 'mdp' and a arbitarily small positive number 'epsilon'.
"""
function value_iteration(mdp::Humancycle; ϵ::Float64=0.001)
    # initialise utility to zero for all states
    U_prime = Dict(Pair(state, 0.0) for state in mdp.states)
    while true
        U = copy(U_prime)
        δ = 0.0
        # go over all possible state
        for state in mdp.states
            # possible outcomes: Array[ Pair(a => [(T(s,a,s', s'), (T(s,a,s', s')]) ]
            outcomes = [Pair(action,transition_model(mdp, state, action)) for action in actions(mdp, state)]
            expectations = map(x-> sum( T*(reward(mdp, state, x.first, ns) + mdp.gamma * U[ns]) for (T,ns) in x.second), outcomes)
            U_prime[state] = maximum(expectations)
            δ = max(δ, abs(U_prime[state] - U[state]));
        end
        if (δ < ((ϵ * (1 - mdp.gamma))/mdp.gamma))
            return U_prime;
        end
    end
end


function policy_extraction(mdp::Humancycle, U::Dict)
	pi = Dict()
	for state in mdp.states
        # possible outcomes: Array[ Pair(a => [(T(s,a,s', s'), (T(s,a,s', s')]) ]
        outcomes = [Pair(action,transition_model(mdp, state, action)) for action in actions(mdp, state)]
        expectations = map(x-> sum( T*(reward(mdp, state, x.first, ns) + mdp.gamma * U[ns]) for (T,ns) in x.second), outcomes)
        @info """\n$(state):\n\n$(outcomes)\n\n$(expectations)"""
		#A = collect(actions(mdp, state))
		#expected_utility = map(x -> sum(p*U[newstate] for (p, newstate) in x),  [transition_model(mdp, state, action) for action in A])
		pi[state] = outcomes[argmax(expectations)].first
	end
	return pi
end



# main script
h = Humancycle(γ=1/40)
U = value_iteration(h)
policy_extraction(h, U)

#=
actions(h,s)
U_prime = Dict(Pair(state, 0.0) for state in h.states)
U = copy(U_prime)
outcomes = [Pair(action,transition_model(h, s, action)) for action in actions(h, s)]
print(outcomes)
map(x-> sum( T*(reward(h, s, x.first, ns) + h.gamma * U[ns]) for (T,ns) in x.second), outcomes)

=#