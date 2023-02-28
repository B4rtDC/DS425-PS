### A Pluto.jl notebook ###
# v0.19.22

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

# ╔═╡ 12e56576-0819-4c6c-8c97-daaba71c9f54
"""
	ingredients(path::String)

Helper function to reload external code. The external file is processed and
returned as a module. 

### Example
```
# load a external file
mymodule = ingredients("path/to/external_file.jl")
# use the external functions
mymodule.foo(args)
```
"""
function ingredients(path::String)
	name = Symbol(basename(path))
	m = Module(name)
	Core.eval(m,
        Expr(:toplevel,
             :(eval(x) = $(Expr(:core, :eval))($name, x)),
             :(include(x) = $(Expr(:top, :include))($name, x)),
             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
             :(include($path))))
	return m

end; nothing

# ╔═╡ 3748807e-28d1-41f2-90b4-e3af784c8ee4
md"""
# Reinforcement learning
In this session, we will explore various techniques for resolving reinforcement learning challenges, using the 4x3 grid world that was previously introduced as a visual demonstration of the concepts involved. We will focus on the model-free part of the tree.

$(PlutoUI.LocalResource("./img/RL_overview.png", :width => 700, :align => "middle"))

Requirements:
- A simulated environment for our agent to interact with (MDP, cf. previous session) that gives rewards based on the actions the agent takes
- Training data to “learn” from: we will need our agent to interact with the environment. We will obtain the data by letting our agent experience multiple trials in the environment.
- An algorithm to train the agent with its hyperparameters (learning rate, discounting factor, ``\dots``)
- Evaluation metric: we will need to define evaluation metrics to measure the performance of our agent. This could include metrics like average reward, success rate, or time to completion.
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
# Passive reinforcement learning
In passive Reinforcement Learning the agent follows a fixed policy ``\pi``. Passive learning attempts to evaluate the given policy  ``\pi`` without any knowledge of the reward function ``R(s, \pi(s), s')`` and the transition model ``T(s,a,s')``


This is usually done by some method of utility estimation. The agent attempts to directly learn the utility of each state that would result from following the policy. At each step, the agent has to perceive the reward and the state - it has no global knowledge of these.

*Note*: if the entire set of actions offers a very low probability of attaining some state ``s^+``, the agent may never perceive the reward ``R(s, \pi(s), s^+)``

The expected utility of a state ``s`` under a policy ``\pi`` can be written as follows:
```math
V^{\pi}(s) = E \left[\sum_{t=0}^{\infty} \gamma^t R(s_t, \pi(s_t), s_{t+1}) \right]
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
V^{\pi}(s) = (1 - \alpha)V^{\pi}(s) + \alpha \underbrace{\left[ R(s, \pi(s),s') + \gamma V^{\pi}(s') \right]}_{\text{``sample''}}
```
or
```math
V^{\pi}(s) = V^{\pi}(s) + \alpha \left[ \underbrace{R(s, \pi(s),s') + \gamma V^{\pi}(s')}_{\text{``sample''}}  - V^{\pi}(s) \right]
```
In the above equationn $\alpha$ is the so-called learning rate. This can be a fixed value or a function which decreases as the number of times a state has been visited increases. In the latter case, better convergence tends to be observed.

The advantage of the temporal difference learning model is its relatively simple computation at each step, rather than having to keep track of various counts.

"""

# ╔═╡ 4b793eb0-846d-45d1-b7d6-7d911006d52a
md"""
## Illustration: 4x3 world
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
## Computer implementation
To be able to use one of the above methods, we need some components:
* take a single action
  ```Julia
  function take_single_action(mdp, s, a)
  	return newstate
  end
  ```
* run a single trial
  ```Julia
  function single_trial!(rlm, mdp)
  	s = initialise_mdp
  	while not finished
  		r = get_reward
  		a = get_action_from_policy
  		learn!(rlm, s, r)  		
  		update_state: s = take_action(mdp, s, a)
  	end
  end
  ```
* a way of applying the RL method and updating it from the trial
  ```Julia
  function learn!(rlm, mdp)
  	update_utility_estimates
  	return action
  end
  ```

You can find some implementations for these methods in the supporting file `PS03_RL_grid_spt.jl`. We reuse the 4x3 gridworld implementation and we added definitions for different reinforcement learning methods.
"""

# ╔═╡ 0cf60950-d3fc-4625-98cf-af88023cc58c
RLgrid = ingredients("./PS03_RL_grid_spt.jl");nothing

# ╔═╡ a4707dfc-8a73-4aa9-8143-dae254bcc84b
md"""
## Application - 4x3 grid world
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
	myMDP = RLgrid.GridMarkovDecisionProcess(initial, terminal_states, grid)
	# policy creation (fixed for passive RL)
	mypolicy = Dict(  (1, 1)=>(1, 0), (1, 2)=>(0, 1), (1, 3)=>(1, 0), (1, 4)=>(0, -1),
					(2, 1)=>(1, 0), 				(2, 3)=>(1, 0), (2, 4)=>nothing,
					(3, 1)=>(0, 1), (3, 2)=>(0, 1),	(3, 3)=>(0, 1), (3, 4)=>nothing) 
	nothing
end

# ╔═╡ 10f9d3b6-c25d-4fe7-9c6f-e27476bad683
md"""
Small demo to illustrate that the results of taking a single action match with the model described in the lectures.
"""

# ╔═╡ 875fb77b-8d2d-48fb-8ce4-970f46870371
counter([RLgrid.take_single_action(myMDP, (1,1),(0,1)) for _ = 1:1000])

# ╔═╡ 66436f39-11b8-4c14-9dcd-ac9b4f3fed10
md"""
### Direct evaluation
Apply direct evaluation (using ``\gamma=1``) to evaluate the utility of eachs state.
"""

# ╔═╡ cd8014db-8776-4f51-9e83-5cd39fe03aad
begin
	# define a learner method with the defined policy
	learner = RLgrid.DirectUtilityEstimation(mypolicy);
	# use a single trial of myMDP to update the learner
	RLgrid.single_trial!(learner, myMDP)
	# show the learned utilities
	println("First trial utilities:\n", join(["$(state): $(val)" for (state,val) in learner.U],"\n"))
	# run another trial
	RLgrid.single_trial!(learner, myMDP)
	# show the learned utilities
	println("\nSecond trial utilities:\n", join(["$(state): $(val)" for (state,val) in learner.U],"\n"))
end

# ╔═╡ baf12281-b3fc-4e11-81cb-51f19cad0f8b
begin
	# run a larger number of trials
	intensive_learner = RLgrid.DirectUtilityEstimation(mypolicy)
	for _ = 1:10000
		RLgrid.single_trial!(intensive_learner, myMDP)
	end
	# show the learned utilities
	println("Final utilities:\n", join(["$(state): $(val)" for (state,val) in intensive_learner.U],"\n"))
end

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

# ╔═╡ b0a6ff6e-1c8d-4cbe-919b-db0881f077b4
begin 
	sample_learner = RLgrid.SampleBasedEstimation(mypolicy, myMDP)
	# run a single trial
	RLgrid.single_trial!(sample_learner, myMDP)
	# show learned transition model
	println("Single trial transition model: (s,a) -> (s', T(s,a,s')) ")
	println(join(["""($(k[1]), $(k[2])) -> $(join(["$(ns), $(nsp))" for (ns,nsp) in v],"; "))""" for (k,v) in sample_learner.mdp.transitions],"\n"))
	# run anothoer trial
	RLgrid.single_trial!(sample_learner, myMDP)
	println("\nSecond trial transition model: (s,a) -> (s', T(s,a,s')) ")
	println(join(["""($(k[1]), $(k[2])) -> $(join(["$(ns), $(nsp))" for (ns,nsp) in v],"; "))""" for (k,v) in sample_learner.mdp.transitions],"\n"))
end

# ╔═╡ ee169a77-5f89-4b63-9003-a16c6ea1aa81
begin 
	# run a larger number of trials
	for _ = 1:500
		RLgrid.single_trial!(sample_learner, myMDP)
	end
	println("Multiple trials transition model: (s,a) -> (s', T(s,a,s')) ")
	println(join(["""($(k[1]), $(k[2])) -> $(join(["$(ns), $(nsp))" for (ns,nsp) in v],"; "))""" for (k,v) in sample_learner.mdp.transitions],"\n"))
end

# ╔═╡ b490cc0c-adde-4a92-8eb7-1178a597ee32
begin
	# show the learned utilities
	println("Final utilities from sample-based policy evaluation:\n", join(["$(state): $(val)" for (state,val) in sample_learner.U],"\n"))
end

# ╔═╡ b0c6e8c7-1439-48d8-bd3d-0de416cfa38a
md"""
### Temporal difference learning
Now we have seen two examples of passive reinforcement learning. The last one is temporal difference learning. Implement this yourself and evaluate it works the 4x3 grid world.
"""

# ╔═╡ 1d8f9a10-fec0-4da1-97ef-de50157ea854


# ╔═╡ 8cc95515-3e55-453a-b113-d66d7ee2f180
md"""
# Active reinforcement learning
We leave our fixed policy behind and allow the agent to decide what actions to take. First you learn the transition model. From this, you know all possible actions. With this information we can use value or policy iteration to find the optimal policy (cf. previous session).
"""

# ╔═╡ c47aff08-2608-4fd6-93df-49204bcf4cc8
md"""
## Q-learning
We avoid the need to even learn a model by learning an action utility function ``Q(s,a)`` instead of ``U(s)``. We use the following update rule:
```math
Q(s,a)_{new} = Q(s,a) + \alpha \left[R(s,a,s') + \gamma \max_{a'}Q(s',a') - Q(s,a) \right]
```
"""

# ╔═╡ 3aa905e6-1dfe-42c7-8ff7-864250079aad
md"""
### Exploration functions
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

# ╔═╡ 83f7dc30-0179-430b-99fd-6dcab375ff8c
md"""
## Implementation
You can find an implementation for Q-learning and some exploration methods in the supporting file `PS03_RL_grid_spt.jl`.
"""

# ╔═╡ 6a1f3aac-1ce3-4f55-80a9-bb9588cd13b2
md"""
## Application - 4x3 grid world
"""

# ╔═╡ 7bde24d2-5cc6-4603-b2ef-450232375124
begin
	# define learner
	myQlearner = RLgrid.Qlearner(myMDP, N_e= 20, R⁺=100.)
	myQlearner.γ = 1.
	
	# first step learning from a percept
	s = (1,1)
	a = RLgrid.learn!(myQlearner, s, -0.04) # new action
	println("""First step:\n s: $(s)\n a: $(a)\n Q(s,a):\n $(join(["\t($(s), $(a))" for (s,a) in myQlearner.Q], "\n"))""")
	# go up from starting state
	newstate = RLgrid.take_single_action(myMDP, s, a)
	# learn again
	RLgrid.learn!(myQlearner, newstate, -0.04)
	println("""\nSecond step:\n s:$(newstate)\n a: $(a)\n Q(s,a):\n $(join(["\t($(s), $(a))" for (s,a) in myQlearner.Q], "\n"))""")
end

# ╔═╡ 930f3c76-8b06-46f9-b098-96e54704ec8c
begin
	bigQlearner = RLgrid.Qlearner(myMDP, N_e= 10, R⁺=1., α=RLgrid.other_α_fun)
	bigQlearner.γ = 1.
	for _ = 1:1000
		RLgrid.single_trial!(bigQlearner, myMDP)
	end
	
	println("""\nMultiple iterations:\nQ(s,a):\n $(join(["\t($(s), $(a))" for (s,a) in bigQlearner.Q], "\n"))""")
end

# ╔═╡ f2e5e3da-724a-4de1-9d80-084594edc8bb
let
	# illustration of Q-values in a specific state for decision making
	s = (1,1)
	(a, Q) = argmax(x->x[2], [(a,bigQlearner.Q[(s, a)]) for a in RLgrid.actions(myMDP, s)])
	println("in state $(s), you're best to take action $(a) ($(RLgrid.arrow_characters[a]))")
end

# ╔═╡ 69fa24ed-8dcb-47b1-8d97-3e5882487c2f
md"""
# Application - How to drive a car on a racetrack

## Problem formulation
This application is from the book [Reinforcement learning: an introduction](http://www.incompleteideas.net/book/the-book-2nd.html) by Richard S. Sutton 
and Andrew G. Barto and was tweaked a bit to match this course.


$(PlutoUI.LocalResource("./img/RL_track.png", :width => 200, :align => "center"))

You want to go as fast as possible around the turn, but not so fast as to run off the track. In our simplified racetrack, the car is at one of a discrete set of grid positions, the cells in the diagram. The velocity is also discrete, a number of grid cells moved horizontally and vertically per time step. The actions are increments to the velocity components. Each may be changed by +1, -1, or 0 in each step, for a total of nine (3 x 3) actions. Both velocity components are restricted to be nonnegative and less than 5, and they cannot both be zero except at the starting line. 

Each episode begins in one of the randomly selected start states with both velocity components zero and ends when the car crosses the finish line. The rewards are -1 for each step until the car crosses the finish line. If the car hits the track boundary, it is moved back to a random position on the starting line, both velocity components are reduced to zero, and the episode continues. Before updating the car’s location at each time step, check to see if the projected path of the car intersects the track boundary. If it intersects the finish line, the episode ends; if it intersects anywhere else, the car is considered to have hit the track boundary and is sent back to the starting line.

Think about states, actions and how we can make this fit in our framework. Try to make some illustrations showing: 
* some trajactories
* the effect of the learned policy (i.e. the duration of a run) in function of the number of iterations.
* the effect of the exploration function and its parameters on the performance given a preset training time
* ``\dots``
"""

# ╔═╡ 1ad8d9af-5be6-4ffc-bcfb-db906bddcd18
md"""
## Implementation
Some implementations are given in the supporting file `PS03_RL_car_spt.jl`.
"""

# ╔═╡ 2ad58812-129f-4376-848a-115d6334ddb0
RLcar = ingredients("./PS03_RL_car_spt.jl");nothing

# ╔═╡ 3592021b-d5ea-4800-bd86-6f82112386d2
begin
	# small demo usage
	# obtain the MDP linked to the car
	model = RLcar.CarMDP(κ=0.1)
	# get a starting state
	state = RLcar.initial_state(model)
	# get the available action in the starting state
	available_actions = RLcar.actions(model, state)
	println("- mdp: $(model)\n- current state: $(state)\n- available actions:\n$(join(["\t$(a)" for a in available_actions],"\n"))")
end

# ╔═╡ 12bc3fe1-0a48-4771-82f7-a81429d677fd
begin
	# drive the car in a random fashion: episode takes up to 1000 moves before stopping
	randompath = RLcar.random_driver(model, episode_length=1000)
	RLcar.plot_track(model, randompath)
end

# ╔═╡ f6f64017-a467-45b9-a0a9-7f607ebcabb4
md"""
## Exploration functions
In the small module, four different exploration functions are provided:
- `IdentityExplorationMethod`: aka “greedy exploration”, picks the largest Q-value
- `RandomExplorationMethod`: picks random actions, disregards the Q-values
- `SimpleExplorationMethod`: rewards a value R⁺ for a state-action pair seen less than `N_max` times, otherwise returns the Q-value
- `EpsilonGreedyExplorationMethod`: picks the largest Q-value, except for a small probability of taking a random action

Do you understand how these work?
"""

# ╔═╡ 0e7ea273-3c8e-41bf-ace4-16b41fbbce55
let
	# identity exploration object
	explorer = RLcar.IdentityExplorationMethod()
	# return an action given the Q-values (here an empty dict), the state and the available actions
	new_action = explorer.exploration_function(Dict(), state, available_actions)
	println("method:\n\t $(explorer)\nnewaction:\n\t $(new_action)")
end

# ╔═╡ 13f616f2-266c-4524-97d8-07d93cf7c1bb
let
	## random exploration
	explorer = RLcar.RandomExplorationMethod()
	# return an action given the Q-values (here an empty dict), the state and the available actions
	new_action = explorer.exploration_function(Dict(), state, available_actions)
	println("method:\n\t $(explorer)\nnewaction:\n\t $(new_action)")
end

# ╔═╡ 6e8e3cf9-a37e-4ec7-999b-3a7b11e77be0
let
	## count-based exploration given a reward of 20 when a state has been seen less than 10 times
	explorer = RLcar.SimpleExplorationMethod(20., 10)
	# return an action given the Q-values (here an empty dict), the state and the available actions
	new_action = explorer.exploration_function(Dict(), state, available_actions)
	explorer, explorer.N, new_action 
	println("""method:\n\t $(explorer)\nnewaction:\n\t $(new_action)\nobserved counts (s,a) => N:\n$(join(["\t$(k) => $(n)" for (k,n) in explorer.N],"\n"))""")
end

# ╔═╡ e9dd19bf-dd77-42a0-8e7d-088749920b36
let
	# ϵ-greedy exploration
	explorer = RLcar.EpsilonGreedyExplorationMethod(0.05)
	# return an action given the Q-values (here an empty dict), the state and the available actions
	new_action = explorer.exploration_function(Dict(), state, available_actions)
	println("method:\n\t $(explorer)\nnewaction:\n\t $(new_action)")
end

# ╔═╡ db6408a0-7695-456a-a367-1f0e26588c8b
md"""
## Learning to drive
We now have the necessary tools to learn from different trials and show the outcome.
"""

# ╔═╡ cd4a4df1-96a1-412a-9992-45cbbddc1a13
begin
	# define exploration method
	explorer = RLcar.EpsilonGreedyExplorationMethod(0.05)
	# define learning method
	carlearner = RLcar.Qlearner(model, explorer, trial_length=1000, N_episodes=10000)
end

# ╔═╡ 35a0b254-4517-41a2-9830-c6c358ae8131
# train from examples
RLcar.learn!(carlearner)

# ╔═╡ bc1cac56-0012-4258-8cd7-57bc66e726f3
# generate a random path for trained model
RLcar.plot_track(model, RLcar.sample(carlearner, initial_state=(1,6,0,0), max_duration=1000))

# ╔═╡ 8959f487-29b7-4762-abf5-36159b6899bf
md"""
## Hyperparameter study
Evaluate the performance of out model and the different methods by selection a suitable performance metric.
"""

# ╔═╡ 65c6db8f-3e88-4214-84a0-7c0c4e7299d5
# to do

# ╔═╡ Cell order:
# ╟─3655d9d6-aeb9-4242-ac20-983ca65b3ab4
# ╟─22df4d74-cfb0-11ec-197e-5b598bde61ca
# ╟─12e56576-0819-4c6c-8c97-daaba71c9f54
# ╟─3748807e-28d1-41f2-90b4-e3af784c8ee4
# ╟─7e0063db-9c82-425b-8cc0-62f81d51e6ce
# ╟─8cb45f23-3e60-424f-a9c9-1974b26237e9
# ╟─1be47c17-54f1-4167-b39b-83a83c0f4cca
# ╟─3ba45299-0869-4ad6-83d1-ed724c546661
# ╟─bc8de56a-85dd-463d-9ba7-7816054a3974
# ╟─0f7fc47d-62cd-45e6-944b-7185bc014c1a
# ╟─4b793eb0-846d-45d1-b7d6-7d911006d52a
# ╟─7de388ed-1cbc-43f7-a421-b4ca3c462c82
# ╠═0cf60950-d3fc-4625-98cf-af88023cc58c
# ╟─a4707dfc-8a73-4aa9-8143-dae254bcc84b
# ╠═f996faa8-7413-4a92-b270-f00252fa8c94
# ╟─10f9d3b6-c25d-4fe7-9c6f-e27476bad683
# ╠═875fb77b-8d2d-48fb-8ce4-970f46870371
# ╟─66436f39-11b8-4c14-9dcd-ac9b4f3fed10
# ╠═cd8014db-8776-4f51-9e83-5cd39fe03aad
# ╠═baf12281-b3fc-4e11-81cb-51f19cad0f8b
# ╟─1bc63c43-a218-4c11-ab38-e6d3d65dc7cf
# ╠═6af33788-ba86-48c1-88a8-57604312130e
# ╟─1807328b-653d-404f-9e39-49ca5b01f55f
# ╠═b0a6ff6e-1c8d-4cbe-919b-db0881f077b4
# ╠═ee169a77-5f89-4b63-9003-a16c6ea1aa81
# ╟─b490cc0c-adde-4a92-8eb7-1178a597ee32
# ╟─b0c6e8c7-1439-48d8-bd3d-0de416cfa38a
# ╠═1d8f9a10-fec0-4da1-97ef-de50157ea854
# ╟─8cc95515-3e55-453a-b113-d66d7ee2f180
# ╟─c47aff08-2608-4fd6-93df-49204bcf4cc8
# ╟─3aa905e6-1dfe-42c7-8ff7-864250079aad
# ╟─83f7dc30-0179-430b-99fd-6dcab375ff8c
# ╟─6a1f3aac-1ce3-4f55-80a9-bb9588cd13b2
# ╠═7bde24d2-5cc6-4603-b2ef-450232375124
# ╠═930f3c76-8b06-46f9-b098-96e54704ec8c
# ╠═f2e5e3da-724a-4de1-9d80-084594edc8bb
# ╟─69fa24ed-8dcb-47b1-8d97-3e5882487c2f
# ╟─1ad8d9af-5be6-4ffc-bcfb-db906bddcd18
# ╠═2ad58812-129f-4376-848a-115d6334ddb0
# ╠═3592021b-d5ea-4800-bd86-6f82112386d2
# ╠═12bc3fe1-0a48-4771-82f7-a81429d677fd
# ╟─f6f64017-a467-45b9-a0a9-7f607ebcabb4
# ╠═0e7ea273-3c8e-41bf-ace4-16b41fbbce55
# ╠═13f616f2-266c-4524-97d8-07d93cf7c1bb
# ╠═6e8e3cf9-a37e-4ec7-999b-3a7b11e77be0
# ╠═e9dd19bf-dd77-42a0-8e7d-088749920b36
# ╟─db6408a0-7695-456a-a367-1f0e26588c8b
# ╠═cd4a4df1-96a1-412a-9992-45cbbddc1a13
# ╠═35a0b254-4517-41a2-9830-c6c358ae8131
# ╠═bc1cac56-0012-4258-8cd7-57bc66e726f3
# ╟─8959f487-29b7-4762-abf5-36159b6899bf
# ╠═65c6db8f-3e88-4214-84a0-7c0c4e7299d5
