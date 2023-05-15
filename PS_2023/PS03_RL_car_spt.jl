
using StatsBase
using Plots


abstract type AbstractMarkovDecisionProcess end
# ------------------------------------------------------
# Car Problem MDP formulation
# ------------------------------------------------------
# actions can be horizontal and vertical velocity changes with value 0, 1 or -1, represented by 1 to 9
const all_car_actions = [(0, 0),(0, 1),(0, -1),(1, 0),(1, 1),(1, -1),(-1, 0),(-1, 1),(-1, -1)]

"""
    CarMDP(;γ=0.9 κ=0.0, number_of_velocities=5, r_finish=100.0)

A Markov Decision Process (MDP) for a car driving on a track.
- The discounting factor is `γ`.
- The state is represented by the position (x,y) and velocity (vx, vy).
- The action is represented by the velocity change (dvx, dvy).
- The reward is negative (-1) for each timestep and positive (+100) for crossing the finish line.
- The transition model is stochastic, with probability `κ` of not reacting to the accelerator.
- The track is represented by a matrix of 0s and 1s, where 1s are obstacles.
- The start line is the first row of the matrix. For each state, the valid actions are precomputed in the following way=
    - the velocity components are restricted to be nonnegative (this is a significant constraint, we cannot move left or back!)
    - less than `number_of_velocities` (<5 in case of the default value)
    - the values or not zero simultaneously
The velocities are discrete values from 0 to `number_of_velocities`-1.
Performance metric is the time in the game loop.

See also: `transition_model`, `reward`, `actions`, `initial_state`, `plot_track`, `plot_track_terminal`
"""
struct CarMDP <: AbstractMarkovDecisionProcess
    γ::Float64
    track # matrix holding the track
    valid_actions # valid actions for each velocity combination
    start_cols # start line columns
    fin_cells # finish line cells
    κ # probability of not reacting to the accelerator
    r_finish # reward for finishing
    #t # duration in the game loop (used as a stop condition)
    number_of_velocities # number of discrete velocities (used for valid actions)

    function CarMDP(;γ=0.9, κ=0.0, number_of_velocities=5, r_finish=100.0)
        (0 < γ <= 1) ? nothing : error("MarkovDecisionProcess():\nThe γ variable of an MDP must be between 0 and 1, the constructor was given ", γ, "!")
        # initiate the track
        rows, cols = 32, 17
        track = zeros(Int8, rows, cols)
        track[32, 1:3] .= 1
        track[31, 1:2] .= 1
        track[30, 1:2] .= 1
        track[29, 1] = 1
        track[1:18, 1] .= 1
        track[1:10, 2] .= 1
        track[1:3, 3] .= 1
        track[1:26, 10:end] .= 1
        track[26, 10] = 0

        # track markers
        start_cols = 4:9  # start line columns
        fin_cells = Set([(27, cols),(28, cols),(29, cols),(30, cols),(31, cols),(32, cols),])  # finish cells

        # establish valid actions for each position x velocity profiles
        valid_actions = Dict((h,v) => filter(dv ->  h+dv[1] ∈ 0:number_of_velocities-1 && 
                                                    v+dv[2] ∈ 0:number_of_velocities-1 && 
                                                    !(h+dv[1] == 0 && h+dv[2] == 0), all_car_actions) for h = 0:number_of_velocities-1, v = 0:number_of_velocities-1)
        # random starting position
        #start_state = (1, rand(start_cols), 0, 0)
        #return new(start_state, track, valid_actions, start_cols, fin_cells, κ, 0, number_of_velocities)
        return new(γ, track, valid_actions, start_cols, fin_cells, κ, r_finish, number_of_velocities)
    end
end

Base.show(io::IO, m::CarMDP) = print(io, "CarMDP with $(m.number_of_velocities) discrete velocities (γ = $(m.γ), κ = $(m.κ), r_finish = $(m.r_finish))")


"""
    transition_model(m::CarMDP, s, a)

From a state and an action, obtain a new state accounting for accelerator sometimes not working.
Returns an array of possible next states, with their probabilities.
"""
function transition_model(m::CarMDP, s, a)
    # from a state and an action, obtain a new state (s, a) -> s' accounting for accelerator sometimes not working.
    new_vel = s[3:4] .+ a
    new_pos = s[1:2] .+ s[3:4]

    # if we finish, return nothing, as there is no future state
    # trace the path from the current state to the next state
    path = Set([    (    min(s[1] + i, new_pos[1]),     min(s[2] + i, new_pos[2])    )    for i = 0:maximum(s[3:4])]) 
    if !isempty(intersect(path, m.fin_cells))
        @debug "$(m.t) crossed the finish line"
        newstate = nothing
    
    # if we go out of bounds, reset to start position # out of bound = index !checkbounds(track or) new state
    elseif !checkbounds(Bool, m.track, new_pos[1:2]...) || m.track[new_pos[1:2]...] == 1
        @debug "$(m.t) - out of bounds, reverting to start position"
        newstate =  (1, rand(m.start_cols), 0, 0)
    
    # otherwise just update
    else
        newstate = (new_pos..., new_vel...)
    end
    
    return [(m.κ, ((s[1:2] .+ s[3:4])..., s[3:4]...)); (1-m.κ, newstate)]
end

# give the reward for an action
reward(m::CarMDP, s, a, sᶥ) = isnothing(sᶥ) ? m.r_finish  : -1.


"""
    actions(m::CarMDP, state)

From a state, obtain all valid actions.
"""
actions(m::CarMDP, state) = m.valid_actions[state[3:4]...]


# ----------------------------- #
# helper functions for car MDP  #
# ----------------------------- #
"""
    plot_track(m::CarMDP, states::Vector)

Show the track, graphically, with velocity vector(s)
"""
function plot_track(m::CarMDP, states::Vector)
    f = heatmap(m.track, aspect_ratio=1, legend=false, grid=false, 
        xticks=false, yticks=false, colorbar=false, axis=false,
        xlims=(0.5, size(m.track, 2)+0.5), ylims=(0.5, size(m.track, 1)+0.5))
    for s in states
        quiver!([s[2]], [s[1]], quiver=([s[4]], [s[3]]), arrow=true, color=:red)
    end
    return f
end
plot_track(m::CarMDP) = plot_track(m, [])
plot_track(m::CarMDP, state::NTuple) = plot_track(m, [state])


"""
    random_driver(m::CarMDP; episode_length=100)

Run a random driver policy for `episode_length` steps or until completion.
"""
function random_driver(m::CarMDP; episode_length=100)
    duration = 0
    path = [initial_state(m)]
    # run the policy until we finish
    while !isnothing(path[end]) && (duration < episode_length)
        # pick random action
        a = rand(actions(m, path[end]))
        # obtain new state
        ns = take_single_action(m, path[end], a)
        # update score
        duration  += 1
        if isnothing(ns)
            break
        end
        # update MDP
        push!(path, ns)
    end
    
    return path
end


# ------------------------------------------------------
# now we can define the RL related functions
# ------------------------------------------------------

"""
    initial_state(m::CarMDP)

Returns a random starting state for the car.
"""
initial_state(m::CarMDP) = (1, rand(m.start_cols), 0, 0)


"""
    take_single_action(m::CarMDP, s, a)

Obtain the result `sᶥ` of a single action `a` from state `s`, accounting for the underlying transition model `m`.
"""
function take_single_action(m::T, s, a) where T <: AbstractMarkovDecisionProcess
    x = rand() # random number (reference value)
    p_cum = 0. # cumulative probability
    for (p, newstate) in transition_model(m, s, a)
        p_cum += p
        if x < p_cum
            return newstate
        end
    end
end

# ------------------------------------------------------
# General RL exploration methods
# -———————————————--------------------------------------

"""
    ExplorationMethod

Abstract type for exploration methods. Each exploration method must implement a function `exploration_function` 
that takes as input the Q-table, the current state and the set of valid actions.
It returns the action to be taken.
"""
abstract type ExplorationMethod end


"""
    IdentityExplorationMethod()

Identity exploration method. The agent will always take the action that maximizes the Q-value.
"""
struct IdentityExplorationMethod 
    exploration_function::Function
    function IdentityExplorationMethod()
        exploration_function = function(Q, s, actions)
            return argmax(a -> get!(Q,(s, a), 0.), actions)
        end
        return new(exploration_function)
    end
end
Base.show(io::IO, m::IdentityExplorationMethod) = print(io, "IdentityExplorationMethod")


"""
RandomExplorationMethod()

Random exploration method. The agent will take random actions
"""
struct RandomExplorationMethod 
    exploration_function::Function
    function RandomExplorationMethod()
        exploration_function = function(Q, s, actions)
            return rand(actions)
        end
        return new(exploration_function)
    end
end
Base.show(io::IO, m::RandomExplorationMethod) = print(io, "RandomExplorationMethod")


"""
    SimpleExplorationMethod(R⁺, N_max)

Simple exploration method. The agent will receive a value of `R⁺` if it has seen the state-action pair less than `N_max` times,
otherwise it will receive a reward equal to the Q-value.

# Parameters
- `R⁺`: the reward for a new state-action pair if it has been seen less than `N_max` times
- `N_max`: the maximum number of times a state-action pair can be seen before the agent receives the Q-value as reward

# Other fields
- `N`: a dictionary that keeps track of the number of times a state-action pair has been seen
"""
mutable struct SimpleExplorationMethod{T} <: ExplorationMethod where T<:Number
    R⁺::T
    N_max::Int
    N::Dict
    exploration_function::Function

    function SimpleExplorationMethod(R⁺::T, N_max::Int) where T<:Number
        # define the object
        N = Dict()
        method =  new{T}(R⁺, N_max, N, identity)
        
        # function that does the actual computation
        exploration_function = function(Q, s, actions; N=method.N, R⁺=method.R⁺, N_max=method.N_max)
            # update the the counts
            for a in actions
                N[(s,a)] = get!(N, (s,a), 0) + 1
            end

            # return the optimal action
            return argmax(a -> get!(N,(s, a), 0.) < N_max ? R⁺ : get!(Q,(s, a), 0.), actions)
        end

        method.exploration_function = exploration_function

        return method
    end
end
Base.show(io::IO, m::SimpleExplorationMethod) = print(io, "SimpleExplorationMethod (R⁺ = $(m.R⁺), N_max = $(m.N_max))")


"""
    EpsilonGreedyExplorationMethod(epsilon::Float64)

Epsilon-greedy exploration method. With probability `epsilon`, the agent will explore, otherwise it will exploit.
"""
struct EpsilonGreedyExplorationMethod <: ExplorationMethod
    ϵ::Float64
    exploration_function::Function
    function EpsilonGreedyExplorationMethod(ϵ::Float64=0.1)
        # function that does the actual computation
        exploration_function = function(Q, s, actions)
            if rand() < ϵ
                return rand(actions)
            else
                return argmax(a -> get!(Q,(s, a), 0.), actions)
            end
        end
    
        return new(ϵ, exploration_function)
    end
end
Base.show(io::IO, m::ExplorationMethod) = print(io, "EpsilonGreedyExplorationMethod with ϵ = $(m.ϵ)")


# ------------------------------------------------------
# General RL methods
# -———————————————--------------------------------------
abstract type ReinforcementLearningMethod end

"""
    Qlearner(mdp::T) where T <: AbstractMarkovDecisionProcess

Q-learning algorithm.

# Parameters
- `mdp`: the underlying Markov Decision Proces, a from of an `AbstractMarkovDecisionProcess`
- `explorer`: the `ExplorationMethod` that will be used to select the next action (exploration should deminish over time, to avoid "trashing")
- `α`: the learning rate for the Q-learning algorithm (should be between 0 an 1 and decrease over time)
- `N_episodes`: the number of episodes to run
- `trial_length`: the maximum number of steps in each episode

# Other fields
- `Q`: the Q-table
- `state`: the current state
- `action`: the current action
- `reward`: the current reward

"""
mutable struct Qlearner{T,M} <: ReinforcementLearningMethod where {T<:AbstractMarkovDecisionProcess,M<:ExplorationMethod}
    mdp::T
    exploration_method::M
    α::Float64
    Q::Dict
    N_episodes::Int64
    trial_length::Int64
    state
    action
    reward
    function Qlearner(mdp::T,   exploration_method::M=IdentityExplorationMethod();
                                α::Float64=0.9,
                                N_episodes::Int64=1,
                                trial_length::Int64=100) where {T<:AbstractMarkovDecisionProcess, M<:ExplorationMethod}
        
        return new{T, M}(mdp, exploration_method, α, Dict(), N_episodes, trial_length, nothing, nothing, nothing)
    end
end
Base.show(io::IO, m::Qlearner) = print(io, "Q-learning algorithm for $(m.mdp)\n (using $(m.exploration_method), α = $(m.α), $(m.N_episodes) episodes, trial length = $(m.trial_length))")

function init!(rlm::Qlearner)
    rlm.state = initial_state(rlm.mdp)
    rlm.action = nothing
    rlm.reward = nothing
end

function single_trial!(rlm::V) where V<:ReinforcementLearningMethod
    # initialize the trial
    init!(rlm) # this will set the state, action and reward to nothing
    trial_length = 0
    mdp = rlm.mdp
    s = rlm.state
    Q = rlm.Q
    f = rlm.exploration_method.exploration_function
    # run the trial
    while !isnothing(s) && (trial_length < rlm.trial_length)
        # select the best action according to the Q-table and the exploration method
        a = f(Q, s, actions(mdp, s))
        # select the best action
        #a = argmax(a -> get!(Q,(s,a), 0.), actions(mdp, s))
        # take the action in the world
        sᶥ = take_single_action(mdp, s, a)
        # get the reward
        r = reward(mdp, s, a, sᶥ)
        # update the Q-table
        learn!(rlm, s, a, sᶥ, r)
        # update the state
        s = sᶥ
        # update the trial length
        trial_length += 1
    end
end

"""
    learn!(rlm::Qlearner, s, a, sᶥ, r)

Update the Q-table using the Q-learning algorithm when going from state `s` to state `sᶥ` by action `a` and receiving reward `rᶥ`.
"""
function learn!(rlm::Qlearner, s, a, sᶥ, r)
    mdp = rlm.mdp
    Q = rlm.Q
    α = rlm.α
    γ = rlm.mdp.γ
    
    # check for terminal state
    if isnothing(sᶥ)
        Q[nothing] = r
    else
        # update the Q-table
        sample = r + γ* maximum( get!(Q, (sᶥ, aᶥ), 0.) for aᶥ in actions(mdp, sᶥ))
        Q[(s,a)] = (1-α) * get!(Q, (s,a), 0.) + α * sample
    end
end

"""
    learn!(rlm::T) where T<:ReinforcementLearningMethod

Learn from the experience by using a specific algorithm.
"""
function learn!(rlm::T) where T<:ReinforcementLearningMethod
    for i in 1:rlm.N_episodes
        single_trial!(rlm)
    end
end

function sample(rlm::T; initial_state=nothing, max_duration::Int=1000) where T<:ReinforcementLearningMethod
    model = rlm.mdp
    if isnothing(initial_state)
        s = RLMethods.initial_state(model)
        #s = initial_state(rlm.mdp)
    else
        s = initial_state
    end
    states = [s]
    duration = 0
    while !isnothing(s) && (duration < max_duration)
        s = states[end]
        a = argmax(a -> get!(rlm.Q, (s,a), 0.), actions(rlm.mdp, s))
        sᶥ = take_single_action(rlm.mdp, s, a)
        if isnothing(sᶥ)
            break
        end
        push!(states, sᶥ)
        duration += 1
    end
    return states
end



