### A Pluto.jl notebook ###
# v0.18.4

using Markdown
using InteractiveUtils

# ╔═╡ 22df4d74-cfb0-11ec-197e-5b598bde61ca
begin
	# dependencies
	using Pkg
	Pkg.activate(joinpath(@__DIR__,".."))
	
	using PlutoUI
	

	using Plots
	using DataStructures
	#using Graphs, SimpleWeightedGraphs, LinearAlgebra, Random
	#import StatsBase:sample
	TableOfContents()
end

# ╔═╡ 3655d9d6-aeb9-4242-ac20-983ca65b3ab4
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 2000px;
    	padding-left: max(160px, 10%);
    	padding-right: max(160px, 10%);
	}
</style>
"""

# ╔═╡ 7e0063db-9c82-425b-8cc0-62f81d51e6ce
begin
	abstract type AbstractMarkovDecisionProcess end

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
	        (0 < gamma <= 1) ? nothing : error("MarkovDecisionProcess():\nThe gamma variable of an MDP must be between 0 and 1, the constructor was given ", gamma, "!")
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
	#=
	"""
	    transition_model(mdp::T, state, action) where {T <: AbstractMarkovDecisionProcess}
	
	Return a list of (P(s'|s, a), s') pairs given the state 's' and action 'a'.
	"""
	function transition_model(mdp::T, state, action) where {T <: AbstractMarkovDecisionProcess}
	    length(mdp.transitions) == 0 ? nothing : error("transition_model(): The transition model for the given 'mdp' could not be found!")
	
	    return mdp.transitions[state][action]
	end=#
	
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
	nothing


	"""
	    GridMarkovDecisionProcess is a MDP implementation of the grid from chapter 16. 
	
	Obstacles in the environment are represented by `nothing`.
	
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
	
	nothing

end

# ╔═╡ 8cb45f23-3e60-424f-a9c9-1974b26237e9
md"""
# Reinforcement learning
# Passive reinforcement learning
In passive Reinforcement Learning the agent follows a fixed policy ``\pi``. Passive learning attempts to evaluate the given policy  ``\pi`` without any knowledge of the reward function ``R(s, \pi(s), s')`` and the transition model ``T(s,a,s')``


This is usually done by some method of utility estimation. The agent attempts to directly learn the utility of each state that would result from following the policy. At each step, the agent has to perceive the reward and the state - it has no global knowledge of these.

*Note*: if the entire set of actions offers a very low probability of attaining some state ``s^+``, the agent may never perceive the reward ``R(s, \pi(s), s^+)``

The utility of a state under a policy can be written as follows:
```math
U^{\pi}(s) = E \left[\sum_{t=0}^{\infty} \gamma^t R(s_t, \pi(s_t), s_{t+1}) \right]
```
Different methods to estimate the utility exist.
"""

# ╔═╡ 1be47c17-54f1-4167-b39b-83a83c0f4cca
md"""
## Methods
"""

# ╔═╡ 3ba45299-0869-4ad6-83d1-ed724c546661
md"""
### Direct evaluation
aka “direct utility estimation” (section 23.2.1).

Construct an agent that follows the policy until it reaches the terminal state. At each step, it logs its current state and reward. Once it reaches the terminal state, it can estimate the utility for each state for that iteration, by simply summing the discounted rewards from that state to the terminal one.

If you run this 'trial' multiple times, you can compute the average utility of each state. If a state occurs more than once in a trial, both its utility values are counted separately.

Downsides:
- can be very slow for large statespace
- wastes information about state connections (and thus the transition probability ``T(s,a,s')``
"""

# ╔═╡ bc8de56a-85dd-463d-9ba7-7816054a3974
md"""
### Sample-Based Policy Evaluation

aka “adaptive dynamic programming” (section 23.2.2)

This method estimates the transition probability ``T(s,a,s')`` by counting the new states resulting from previous states and actions. The program runs through the policy a number of times, keeping track of:
- ``N_{s,a}``: number of occurrences of state s and the policy-recommended action a.
- ``N_{s'|s,a}``: number of occurrence of s' resulting from a on s
These two values can be used to estimate the transition model:
```math
\hat{T}(s,a,s') = \frac{N_{s'|s,a}}{N_{s,a}}
```
Once we have an estimate of the transition model, we can apply the policy evaluation algorithm from the previous session on MDPs to estimate the utilities.

Downsides: slow/intractable for large state space
"""

# ╔═╡ 0f7fc47d-62cd-45e6-944b-7185bc014c1a
md"""
### Temporal difference learning
This method uses the difference in utility between succesive state to update the utility, i.e. adjust the utility based on the observed successor. Note the contrast with the previous method, where this was done for all possible successors. (section 23.2.3). 

```math
U^{\pi}(s) = U^{\pi}(s) + \alpha \left[ R(s, \pi(s),s') + \gamma U^{\pi}(s') - U^{\pi}(s)\right]
```
In the above equationn $\alpha$ is the so-called learning rate. This can be a fixed value or a function which decreases as the number of times a state has been visited increases. In the latter case, better convergence tends to be observed.

The advantage of the temporal difference learning model is its relatively simple computation at each step, rather than having to keep track of various counts.

"""

# ╔═╡ 4b793eb0-846d-45d1-b7d6-7d911006d52a
md"""
## Small application: 4x3 world
Apply the different passive reinforcement learning methods to the following sequence of trials from a predefined policy. The policy is shown under the arrows and the reward is shown above the arrows. 
```math
(1,1)\overset{-0.04}{\underset{up}{\rightarrow}} 
(1,2)\overset{-0.04}{\underset{up}{\rightarrow}} 
(1,3)\overset{-0.04}{\underset{right}{\rightarrow}} 
(1,2)\overset{-0.04}{\underset{up}{\rightarrow}}
(1,3)\overset{-0.04}{\underset{right}{\rightarrow}}
(2,3)\overset{-0.04}{\underset{right}{\rightarrow}}
(3,3)\overset{+1}{\underset{right}{\rightarrow}}
(4,3)
```
```math
(1,1)\overset{-0.04}{\underset{up}{\rightarrow}} 
(1,2)\overset{-0.04}{\underset{up}{\rightarrow}} 
(1,3)\overset{-0.04}{\underset{right}{\rightarrow}} 
(2,3)\overset{-0.04}{\underset{right}{\rightarrow}}
(3,3)\overset{-0.04}{\underset{right}{\rightarrow}}
(3,2)\overset{-0.04}{\underset{up}{\rightarrow}}
(3,3)\overset{+1}{\underset{right}{\rightarrow}}
(4,3)
```
```math
(1,1)\overset{-0.04}{\underset{up}{\rightarrow}} 
(1,2)\overset{-0.04}{\underset{up}{\rightarrow}} 
(1,3)\overset{-0.04}{\underset{right}{\rightarrow}} 
(2,3)\overset{-0.04}{\underset{right}{\rightarrow}}
(3,3)\overset{-0.04}{\underset{right}{\rightarrow}}
(3,2)\overset{-1}{\underset{up}{\rightarrow}}
(4,2)
```
"""

# ╔═╡ 7de388ed-1cbc-43f7-a421-b4ca3c462c82
md"""
## Implementation
To be able to use one of the above methods, we need some components:
- take a single action
- run a single trial
- a way of applying the RL method and updating it from the trial

Below you can find some implementations for these steps
"""

# ╔═╡ 98368c86-ef45-474d-b40c-9de06a36034a
md"""
Below you can find a implementation for direct evaluation
"""

# ╔═╡ 981c4907-66d8-44eb-8030-a97ea44a0c55
md"""
Below you can find an implementation for sample based policy evaluation
"""

# ╔═╡ a4707dfc-8a73-4aa9-8143-dae254bcc84b
md"""
## Computer application - 4x3 grid world
Recall the 4x3 world from the MDP session. We will apply different reinforcement learning techniques on this problem.
"""

# ╔═╡ f996faa8-7413-4a92-b270-f00252fa8c94
begin
	# environment setup
	initial = (1, 1)
	terminal_states = Set([(2, 4), (3, 4)])
	grid = [-0.04 -0.04 -0.04 -0.04;
			-0.04 nothing -0.04 -1;
			-0.04 -0.04 -0.04 +1]
	# environment creation
	myMDP = GridMarkovDecisionProcess(initial, terminal_states, grid)
	# policy creation
	mypolicy = Dict(  (1, 1)=>(1, 0), (1, 2)=>(0, 1), (1, 3)=>(1, 0), (1, 4)=>(0, -1),
					(2, 1)=>(1, 0), 				(2, 3)=>(1, 0), (2, 4)=>nothing,
					(3, 1)=>(0, 1), (3, 2)=>(0, 1),	(3, 3)=>(0, 1), (3, 4)=>nothing) 
	nothing
end

# ╔═╡ 10f9d3b6-c25d-4fe7-9c6f-e27476bad683
md"""
Small demo to illustrate that the results of taking a single action match with the model described in the lectures.
"""

# ╔═╡ 66436f39-11b8-4c14-9dcd-ac9b4f3fed10
md"""
### Direct evaluation
Apply direct evaluation (using ``\gamma=1``) to evaluate the utility of eachs state.
"""

# ╔═╡ 1bc63c43-a218-4c11-ab38-e6d3d65dc7cf
md"""
Make a figure that shows the evolution of the estimate in function of the number of iterations (similar to the illustration you can observe in 23.3)
"""

# ╔═╡ 6af33788-ba86-48c1-88a8-57604312130e
begin # to do
	
end

# ╔═╡ 1807328b-653d-404f-9e39-49ca5b01f55f
md"""
### Sample-Based Policy Evaluation
"""

# ╔═╡ b0c6e8c7-1439-48d8-bd3d-0de416cfa38a
md"""
### Temporal difference learning
Now we have seen two examples of passive reinforcement learning. The last one is temporal differnce learning. Implement this yourself and evaluate it works the 4x3 grid world.
"""

# ╔═╡ 1d8f9a10-fec0-4da1-97ef-de50157ea854


# ╔═╡ 8cc95515-3e55-453a-b113-d66d7ee2f180
md"""
# Active reinforcement learning
We leave our fixed policy behind and allow the agent to decide what actions to take. First you learn the transition model. From this, you know all possible actions. With this information we can use value or policy iteration to find the optimal policy (cf. previous session).
"""

# ╔═╡ 3aa905e6-1dfe-42c7-8ff7-864250079aad
md"""
## Exploration functions
An exploration function ``f(u,n)`` determines how preference for high values of the utility u (exploitaton) are traded off against preference for actions that have not been tried often according to their count n (exploration). Example:
```math
f(u,n) = \cases{R^{+} & if $n<N_e$ \\u & otherwise}
```
where ``N_e`` denotes a fixed value and ``R^{+}`` is an optimistic estimate of the best possible reward obtainable in any state.


The exploration function is used to determine the next action. In the case of Q-learning, this becomes:
```math
a = \text{argmax}_{a'}f\left( Q(s', a'), N_{sa}(s',a') \right)
```


"""

# ╔═╡ c47aff08-2608-4fd6-93df-49204bcf4cc8
md"""
## Q-learning
We avoid the need to even learn a model by learning an action utility function ``Q(s,a)`` instead of ``U(s)``. We use the following update rule:
```math
Q(s,a)_{new} = Q(s,a) + \alpha \left[R(s,a,s') + \gamma \max_{a'}Q(s',a') - Q(s,a) \right]
```
"""

# ╔═╡ 83f7dc30-0179-430b-99fd-6dcab375ff8c
md"""
## Implementation
Below you can find an implementation for the Q-learning algorithm.
"""

# ╔═╡ 4a14b6f6-cdb5-4c66-a9f5-30554341c613
begin
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
	nothing
end

# ╔═╡ ad16fccd-c3b2-45ed-a539-26112629e41b
begin
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

	nothing
end

# ╔═╡ 82e5b311-5578-49c9-b74e-a040b7c67a38
begin
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
	
	nothing
end

# ╔═╡ cd8014db-8776-4f51-9e83-5cd39fe03aad
learner = DirectUtilityEstimation(mypolicy);

# ╔═╡ ae1ef180-bdaf-424a-be1e-7a16eaf0d8f8
learner.buffer

# ╔═╡ c8a9c91f-8dd8-42e8-ad20-ea880bb11760
learner.U

# ╔═╡ d95aff7a-db40-412f-8743-a5a1478ca393
learner.buffer

# ╔═╡ b59b3903-6a20-427a-a5a5-8f3b43585131
begin
	mutable struct ADPEstimation <:ReinformentLearningMethod
		"current state"
		state
		"current action"
		action
		"policy used for RL"
		policy::Dict
		"estimate for utilities"
		U::Dict 
		"internal represenation of the MDP we are learning"
		mdp::MarkovDecisionProcess
		"counts of seeing action a in state s"
		N_sa::Dict
		"counts of action a and state s leading to s prime"
		N_s_prime_sa::Dict
	end

	function ADPEstimation(Π::Dict, mdp::T) where {T <: AbstractMarkovDecisionProcess}
		return ADPEstimation(nothing, nothing, Π, Dict(), 
			MarkovDecisionProcess(mdp.initial, mdp.actions, mdp.terminal_states, Dict(); states=nothing, gamma=1.), Dict(), Dict())
	end

	"""
	    transition_model(mdp::PassiveADPAgentMDP, state, action)
	
	Return a list of (P(s'|s, a), s') pairs given the state 's' and action 'a'.
	"""
	function transition_model(rlm::ADPEstimation, state, action)
	    return collect((v, k) for (k, v) in get!(rlm.mdp.transitions, (state, action), Dict()))
	end

	
	"""
	    policy_evaluation!(pi::Dict, U::Dict, mdp::PassiveADPAgentMDP; k::Int64=20)
	
	Return the updated utilities of the MDP's states by applying the modified policy iteration
	algorithm on the given Markov decision process 'mdp', utility function 'U', policy 'pi',
	and number of Bellman updates to use 'k'.
	"""
	function policy_evaluation!(rlm::ADPEstimation; k::Int64=20)
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
		learn!(rlm::ADPEstimation, s, r) 

	Update the utilies by using the state `s` and the associated reward `r`
	"""
	function learn!(rlm::ADPEstimation, s, r)
		push!(rlm.mdp.states, s)

		# check if s is a new state
		if !haskey(rlm.mdp.reward, s)
			rlm.U[s] = r
			rlm.mdp.reward[s] = r
		end
		
		# update our transition probabilities
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

	init!(rlm::ADPEstimation) = nothing
	update!(rlm::ADPEstimation) = nothing

	nothing
end

# ╔═╡ b0a6ff6e-1c8d-4cbe-919b-db0881f077b4
ADPEstimation(mypolicy, myMDP)

# ╔═╡ 875fb77b-8d2d-48fb-8ce4-970f46870371
counter([take_single_action(myMDP, (1,1),(0,1)) for _ = 1:1000])

# ╔═╡ 115afa0c-192c-46a2-9812-8958be700a65
# run a single trial
single_trial!(learner, myMDP)

# ╔═╡ c275c323-e4b7-4901-8889-08c65f55e771
single_trial!(learner, myMDP)

# ╔═╡ baf12281-b3fc-4e11-81cb-51f19cad0f8b
# run multiple trials
begin
	intensive_learner = DirectUtilityEstimation(mypolicy)
	for _ = 1:10000
		single_trial!(intensive_learner, myMDP)
	end
end

# ╔═╡ 4c1a44a3-9408-4e2e-ab9a-486fea8caf91
intensive_learner.U

# ╔═╡ bee9461b-84df-4cf0-8aa3-92b9356b8eda
begin
	# run a single trial
	ADP_learner = ADPEstimation(mypolicy, myMDP)
	single_trial!(ADP_learner, myMDP)
end

# ╔═╡ 9cb93527-671a-4c62-8354-180e7817682d
ADP_learner.mdp.transitions

# ╔═╡ b490cc0c-adde-4a92-8eb7-1178a597ee32
ADP_learner.mdp.transitions

# ╔═╡ 07d220fe-8f1d-40fe-a49f-8b59ac533a05
ADP_learner.U

# ╔═╡ ee169a77-5f89-4b63-9003-a16c6ea1aa81
begin 
	for _ = 1:100
		single_trial!(ADP_learner, myMDP)
	end
end

# ╔═╡ 7bde24d2-5cc6-4603-b2ef-450232375124
begin
	# define learner
	myQlearner = Qlearner(myMDP, N_e= 20, R⁺=100.)
	myQlearner.γ = 1.
	
	# first step learning from a percept
	a = learn!(myQlearner, (1,1), -0.04) # new action
	@info a, myQlearner.Q # shows current state
	# go up from starting state
	newstate = take_single_action(myMDP, (1,1), a)
	# learn again
	learn!(myQlearner, newstate, -0.04)
	@warn newstate, myQlearner.Q
end

# ╔═╡ 930f3c76-8b06-46f9-b098-96e54704ec8c
begin
	bigQlearner = Qlearner(myMDP, N_e= 10, R⁺=1., α=other_α_fun)
	bigQlearner.γ = 1.
	for _ = 1:1000
		single_trial!(bigQlearner, myMDP)
	end
	bigQlearner.Q
end

# ╔═╡ f2e5e3da-724a-4de1-9d80-084594edc8bb
[(a,bigQlearner.Q[((1,1), a)]) for a in actions(myMDP, (1,1))]

# ╔═╡ Cell order:
# ╟─3655d9d6-aeb9-4242-ac20-983ca65b3ab4
# ╟─22df4d74-cfb0-11ec-197e-5b598bde61ca
# ╟─7e0063db-9c82-425b-8cc0-62f81d51e6ce
# ╟─8cb45f23-3e60-424f-a9c9-1974b26237e9
# ╟─1be47c17-54f1-4167-b39b-83a83c0f4cca
# ╟─3ba45299-0869-4ad6-83d1-ed724c546661
# ╟─bc8de56a-85dd-463d-9ba7-7816054a3974
# ╟─0f7fc47d-62cd-45e6-944b-7185bc014c1a
# ╟─4b793eb0-846d-45d1-b7d6-7d911006d52a
# ╟─7de388ed-1cbc-43f7-a421-b4ca3c462c82
# ╠═ad16fccd-c3b2-45ed-a539-26112629e41b
# ╟─98368c86-ef45-474d-b40c-9de06a36034a
# ╟─82e5b311-5578-49c9-b74e-a040b7c67a38
# ╟─981c4907-66d8-44eb-8030-a97ea44a0c55
# ╠═b59b3903-6a20-427a-a5a5-8f3b43585131
# ╟─a4707dfc-8a73-4aa9-8143-dae254bcc84b
# ╠═f996faa8-7413-4a92-b270-f00252fa8c94
# ╟─10f9d3b6-c25d-4fe7-9c6f-e27476bad683
# ╠═875fb77b-8d2d-48fb-8ce4-970f46870371
# ╟─66436f39-11b8-4c14-9dcd-ac9b4f3fed10
# ╠═cd8014db-8776-4f51-9e83-5cd39fe03aad
# ╠═115afa0c-192c-46a2-9812-8958be700a65
# ╠═ae1ef180-bdaf-424a-be1e-7a16eaf0d8f8
# ╠═c8a9c91f-8dd8-42e8-ad20-ea880bb11760
# ╠═c275c323-e4b7-4901-8889-08c65f55e771
# ╠═d95aff7a-db40-412f-8743-a5a1478ca393
# ╠═baf12281-b3fc-4e11-81cb-51f19cad0f8b
# ╠═4c1a44a3-9408-4e2e-ab9a-486fea8caf91
# ╟─1bc63c43-a218-4c11-ab38-e6d3d65dc7cf
# ╠═6af33788-ba86-48c1-88a8-57604312130e
# ╟─1807328b-653d-404f-9e39-49ca5b01f55f
# ╠═b0a6ff6e-1c8d-4cbe-919b-db0881f077b4
# ╠═bee9461b-84df-4cf0-8aa3-92b9356b8eda
# ╠═9cb93527-671a-4c62-8354-180e7817682d
# ╠═ee169a77-5f89-4b63-9003-a16c6ea1aa81
# ╠═b490cc0c-adde-4a92-8eb7-1178a597ee32
# ╠═07d220fe-8f1d-40fe-a49f-8b59ac533a05
# ╟─b0c6e8c7-1439-48d8-bd3d-0de416cfa38a
# ╠═1d8f9a10-fec0-4da1-97ef-de50157ea854
# ╟─8cc95515-3e55-453a-b113-d66d7ee2f180
# ╟─3aa905e6-1dfe-42c7-8ff7-864250079aad
# ╟─c47aff08-2608-4fd6-93df-49204bcf4cc8
# ╟─83f7dc30-0179-430b-99fd-6dcab375ff8c
# ╠═4a14b6f6-cdb5-4c66-a9f5-30554341c613
# ╠═7bde24d2-5cc6-4603-b2ef-450232375124
# ╠═930f3c76-8b06-46f9-b098-96e54704ec8c
# ╠═f2e5e3da-724a-4de1-9d80-084594edc8bb
