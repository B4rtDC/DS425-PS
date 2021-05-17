abstract type AbstractMarkovDecisionProcess end
## --------------------------------------------- ##
## GENERIC PART
## --------------------------------------------- ##

"""
    MarkovDecisionProcess is a MDP implementation of AbstractMarkovDecisionProcess.

A Markov decision process is a sequential decision problem with fully observable and stochastic environment with a transition model and rewards function.
The discount factor (gamma variable) describes the preference for current rewards over future rewards.
"""
struct MarkovDecisionProcess{T} <: AbstractMarkovDecisionProcess
    initial::T
    states::Set{T}
    actions::Set{T}
    terminal_states::Set{T}
    transitions::Dict
    gamma::Float64
    reward::Dict

    function MarkovDecisionProcess{T}(initial::T, actions_list::Set{T}, terminal_states::Set{T}, transitions::Dict, states::Union{Nothing, Set{T}}, gamma::Float64) where T
        !(0 < gamma <= 1) ? nothing : error("MarkovDecisionProcess():\nThe gamma variable of an MDP must be between 0 and 1, the constructor was given ", gamma, "!")
        new_states = typeof(states) <: Set ? states : Set{typeof(initial)}()

        return new(initial, new_states, actions_list, terminal_states, transitions, gamma, Dict())
    end  
end

function MarkovDecisionProcess(initial, actions_list::Set, terminal_states::Set, transitions::Dict;
                                states::Union{Nothing, Set}=nothing, gamma::Float64=0.9)
    return MarkovDecisionProcess{typeof(initial)}(initial, actions_list, terminal_states, transitions, states, gamma)
end

"""
    reward(mdp::T, state) where {T <: AbstractMarkovDecisionProcess}

Return a reward based on the given 'state'.
"""
function reward(mdp::T, state) where {T <: AbstractMarkovDecisionProcess}
    return mdp.reward[state]
end

"""
    transition_model(mdp::T, state, action) where {T <: AbstractMarkovDecisionProcess}

Return a list of (P(s'|s, a), s') pairs given the state 's' and action 'a'.
"""
function transition_model(mdp::T, state, action) where {T <: AbstractMarkovDecisionProcess}
    length(mdp.transitions) == 0 ? nothing : error("transition_model(): The transition model for the given 'mdp' could not be found!")

    return mdp.transitions[state][action]
end

"""
    actions(mdp::T, state) where {T <: AbstractMarkovDecisionProcess}

Return a set of actions that are possible in the given state.
"""
function actions(mdp::T, state) where {T <: AbstractMarkovDecisionProcess}
    if state in mdp.terminal_states
        return Set{Nothing}([nothing])
    else
        return mdp.actions;
    end
end

## --------------------------------------------- ##
## SPECIFIC PART 4x3 GAME
## --------------------------------------------- ##


"""
    GridMarkovDecisionProcess is a MDP implementation of the grid from chapter 17. 

Obstacles in the environment are representedby a nothing.

Arguments:

"""
struct GridMarkovDecisionProcess <: AbstractMarkovDecisionProcess
    initial::Tuple{Int64, Int64}
    states::Set{Tuple{Int64, Int64}}
    actions::Set{Tuple{Int64, Int64}}
    terminal_states::Set{Tuple{Int64, Int64}}
    grid::Array{Union{Nothing, Float64}, 2}
    gamma::Float64
    reward::Dict
    function GridMarkovDecisionProcess(initial::Tuple{Int64, Int64}, terminal_states::Set{Tuple{Int64, Int64}}, 
                                       grid::Array{Union{Nothing, Float64}, 2}; 
                                       states::Union{Nothing, Set{Tuple{Int64, Int64}}}=nothing, gamma::Float64=0.9)
        (0 < gamma <= 1) ? nothing : error("MarkovDecisionProcess():\nThe gamma variable of an MDP must be between 0 and 1, the constructor was given ", gamma, "!")
        new_states = typeof(states) <: Set ? states : Set{Tuple{Int64, Int64}}()
        orientations::Set = Set{Tuple{Int64, Int64}}([(1, 0), (0, 1), (-1, 0), (0, -1)])
        reward::Dict = Dict()
        for i in 1:size(grid, 1)
            for j in 1:size(grid, 2)
                reward[(i, j)] = grid[i, j]
                if !(grid[i, j] === nothing)
                    push!(new_states, (i, j))
                end
            end
        end

        return new(initial, new_states, orientations, terminal_states, grid, gamma, reward);
    end 
end

"""
    go_to(gmdp::GridMarkovDecisionProcess, state::Tuple{Int64, Int64}, direction::Tuple{Int64, Int64})

Return the next state given the current state and direction. If the next state is not known
"""
function go_to(gmdp::GridMarkovDecisionProcess, state::Tuple{Int64, Int64}, direction::Tuple{Int64, Int64})
    next_state::Tuple{Int64, Int64} = state .+ direction
    
    return next_state in gmdp.states ? next_state : state
end

"""
    transition_model
    
Return an array of (P(s'|s, a), s') pairs given the state 's' and action 'a'.
"""
function transition_model(gmdp::GridMarkovDecisionProcess, state::Tuple{Int64, Int64}, action::Nothing)
    return [(0.0, state)];
end

"""
    transition_model
    
Return an array of (P(s'|s, a), s') pairs given the state 's' and action 'a'.
"""
function transition_model(gmdp::GridMarkovDecisionProcess, state::Tuple{Int64, Int64}, action::Tuple{Int64, Int64})
    return [
            (0.8, go_to(gmdp, state, action)),
            (0.1, go_to(gmdp, state, turn_heading(action, -1))),
            (0.1, go_to(gmdp, state, turn_heading(action, 1)))
            ];
end

# (0, 1) will move the agent rightward.
# (-1, 0) will move the agent upward.
# (0, -1) will move the agent leftward.
# (1, 0) will move the agent downward.
const arrow_characters = Dict((0, 1) => ">", (-1, 0) => "^", (0, -1) => "<", (1, 0) => "v", nothing => ".")

"""
    show_grid(gmdp::GridMarkovDecisionProcess, mapping::Dict)

Returns a string representation of the current grid using [<, >, v, ^] to indication the heading and "." to indicate no action.
"""
function show_grid(gmdp::GridMarkovDecisionProcess, policy::Dict)
    mapping = Dict(state => arrow_characters[action] for (state, action) in policy)
    io = IOBuffer()
    for i in 1:size(gmdp.grid, 1)
        for j in 1:size(gmdp.grid, 2)
            print(io, get(mapping, (i,j), "o"))
        end
        print(io,"\n")
    end
    return String(take!(io))
end


"""
    turn_heading

Given an input heading, return the heading in inc positions further in a clockwise manner 

The headings can be specified with H. The heading h should occur in the collection of headings H.                 
"""
function turn_heading(h::Tuple{Int64,Int64}, inc::Int64; H=[(0,1),(1,0),(0,-1),(-1,0)])
    # get index of current heading
    i = findfirst(x->x==h, H)
    # return incremented heading
    return H[(i + inc + length(H) - 1) % length(H) + 1]
end


## --------------------------------------------- ##
## Solution methods
## --------------------------------------------- ##

"""
    value_iteration(mdp::T; epsilon::Float64=0.001) where {T <: AbstractMarkovDecisionProcess}

Return the utilities of the MDP's states as a Dict by applying the value iteration algorithm (Fig. 17.4)
on the given Markov decision process 'mdp' and a arbitarily small positive number 'epsilon'.
"""
function value_iteration(mdp::T; ϵ::Float64=0.001) where {T <: AbstractMarkovDecisionProcess}
    # initialise utility to zero for all states
    U_prime = Dict(collect(Pair(state, 0.0) for state in mdp.states))
    while true
        U = copy(U_prime)
        δ = 0.0
        for state in mdp.states
            U_prime[state] = reward(mdp, state) + mdp.gamma * maximum(map(x -> sum(p*U[newstate] for (p, newstate) in x), 
                                                                          [transition_model(mdp, state, action) for action in actions(mdp, state)]))
            δ = max(δ, abs(U_prime[state] - U[state]));
        end
        if (δ < ((ϵ * (1 - mdp.gamma))/mdp.gamma))
            return U_prime;
        end
    end
end

function policy_extraction(mdp::T, U::Dict) where T<:AbstractMarkovDecisionProcess
	pi = Dict()
	for state in mdp.states
		A = collect(actions(mdp, state))
		expected_utility = map(x -> sum(p*U[newstate] for (p, newstate) in x),  [transition_model(mdp, state, action) for action in A])
		pi[state] = A[argmax(expected_utility)]
	end
	return pi
end


#function expected_utility(mdp::T, U::Dict, state::Tuple{Int64, Int64}, action::Nothing) where {T <: AbstractMarkovDecisionProcess}
#    return sum((p * U[state_prime] for (p, state_prime) in transition_model(mdp, state, action)));
#end

"""
    policy_evaluation(pi::Dict, U::Dict, mdp::T; k::Int64=20) where {T <: AbstractMarkovDecisionProcess}

Return the updated utilities of the MDP's states by applying the modified policy iteration
algorithm on the given Markov decision process 'mdp', utility function 'U', policy 'pi',
and number of Bellman updates to use 'k'.
"""
function policy_evaluation(pi::Dict, U::Dict, mdp::T; k::Int64=20) where {T <: AbstractMarkovDecisionProcess}
    for i in 1:k
        for state in mdp.states
            U[state] = reward(mdp, state) + mdp.gamma * sum(p * U[newstate] for (p, newstate) in transition_model(mdp, state, pi[state]));
        end
    end
    return U;
end

function policy_evaluation(pi::Dict, U::Dict, gmdp::GridMarkovDecisionProcess; k::Int64=20)
    for i in 1:k
        for state in gmdp.states
            U[state] = reward(gmdp, state) + gmdp.gamma * sum(p * U[newstate] for (p, newstate) in transition_model(gmdp, state, pi[state]));
        end
    end
    return U;
end


"""
    policy_iteration(mdp::T) where {T <: AbstractMarkovDecisionProcess}

Return a policy using the policy iteration algorithm (Fig. 17.7) given the Markov decision process 'mdp'.

it initialises the policy at random
"""
function policy_iteration(mdp::T) where {T <: AbstractMarkovDecisionProcess}
    U = Dict(Pair(state, 0.0) for state in mdp.states)
    pi = Dict(Pair(state, rand(actions(mdp, state))) for state in mdp.states)
    while true
        U = policy_evaluation(pi, U, mdp)
        unchanged = true
        for state in mdp.states
            A = collect(actions(mdp, state))
            expected_utility = map(x -> sum(p*U[newstate] for (p, newstate) in x),  [transition_model(mdp, state, action) for action in A])
            action = A[argmax(expected_utility)]
            if action != pi[state]
                pi[state] = action
                unchanged = false
            end
        end
        if unchanged
            return pi, U;
        end
    end
end

