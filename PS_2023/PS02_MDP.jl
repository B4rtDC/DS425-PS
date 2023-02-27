### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 61e06d99-e14d-462f-a83a-bffd734408f6
begin
	# dependencies
	using Pkg
	Pkg.activate(joinpath(@__DIR__,".."))
	
	using PlutoUI
	

	using Plots
	#using Graphs, SimpleWeightedGraphs, LinearAlgebra, Random
	#import StatsBase:sample
	TableOfContents()
end

# ╔═╡ 91ed838a-d14f-4a0a-863b-4884e5cfa529
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

# ╔═╡ 763946d5-f8b3-430e-b889-20a2559ed15f
md"""
# Markov Decision Process (MDP)
"""

# ╔═╡ 25d705d9-c565-418c-933b-5bd6303e7b24
md"""
## Quick questions
* What are the characteristics of a Markov Decision Process?
* What are the components of a MDP?
* What is a solution in the context of MDPs?
* What is the impact of the discounting factor ``\gamma`` and where does it occur?
* ``\dots`` 
"""

# ╔═╡ 22d811c0-5592-45ef-b542-bb7c3d79c812
md"""
## Small applications
Below we have a set of problem that we want to transform into the MDP formalism. Think about the states, transition probabilties and rewards. Provide the components of the MDP formalism in a graphical way.
1. a cleaning robot needs to collect trash. The robot has limited autonomy. Possible activities include include searching for trash, waiting  or recharging itself. $(@bind robotsolution PlutoUI.CheckBox())
2. a human wants to maximize its earnings. The human can be tired, energetic or healthier. in function of how the human is feeling, he can sleep, work (=earn money) or do a workout. Working always fatigues the humen. Doing a workout makes him feel more energetic when tired and makes him healthier when energetic. $(@bind humansolution PlutoUI.CheckBox())
3. consider the example of the lectures with the robot in the 4x3 grid. provide the MDP formalism. $(@bind lecturesolution PlutoUI.CheckBox())

   $(PlutoUI.LocalResource("./img/MDProbot.png", :width => 300, :align => "center"))

"""

# ╔═╡ 18b033cb-c351-4078-85ec-c7217294a17e
md"""
4. Throughout this course, the pac-man game has been used a lot. Formulate the pac-man game as an MDP. Try to be as complete as possible.
"""

# ╔═╡ eab33b26-56bf-4837-be2a-6b41acf7ab0b
md"""
Below different solutions for the first three questions are shown:

  $(robotsolution ? PlutoUI.LocalResource("./img/cleaningrobotMDP.png", :width => 500, :align => "center") : "")
$(humansolution ? PlutoUI.LocalResource("./img/humanMDP.png", :width => 500, :align => "center") : "")
$(lecturesolution ? PlutoUI.LocalResource("./img/lecturesolution.png", :width => 800, :align => "center") : "")

"""

# ╔═╡ 46e7fb69-8bd0-4a78-a7df-83e71cde2abc
md"""
## Solving an MDP
"""

# ╔═╡ c7c601d8-408e-4227-9c56-0e30f429eb50
md"""
### Value iteration
Starting from the Bellman equation you can solve the $n$ equations for the $n$ states in an iterative fashion.

```math
V_{k+1}(s) = \max_{a}\sum_{s'}T(s,a,s')[R(s,a,s') + \gamma V_k(s') ]
```
where
```math
\begin{eqnarray*}
T(s,a,s')&:&\text{transition probility of going to state $s'$ when doing action $a$ in state $s$}\\
R(s,a,s') &:&\text{reward}\\
\gamma&:&\text{discount factor}
\end{eqnarray*}
```
This algorithm will converge to the unique optimal values $V^{*}(s)$ after a number of iterations after being initialized to zero.

#### Obtaining a policy from value iteration
Once you have the solutions of the problem, you can obtain the optimal policy by doing policy extraction:
```math
\pi^{*}(s)=  \underset{a}{\text{argmax}} \sum_{s'}T(s,a,s')[R(s,a,s') + \gamma V^{*}(s,a,s')]
```
"""

# ╔═╡ 64530d11-46fd-4450-8593-c0521b9a54a6
md"""
#### Value iteration - application
Suppose the value iteration algorithm has converged to the values shown on the figure below. For each state you have the available actions (black dots), the transition model and the rewards. Extract the optimal policy. Use γ = 1/2.

$(PlutoUI.LocalResource("./img/valitsetup.png", :width => 500, :align => "center"))

"""

# ╔═╡ 29226fd6-893e-4ca2-8bfb-6e834a9530bf
md"""
### Policy iteration
In the previous section we have used the non-linear $\max$ operator. If you impose a policy on the system, you no longer have to do this and this reduces the computation from $\mathcal{O}(S^2A)$ to $\mathcal{O}(S^2)$.

We now work in two steps:
1. The update the utilities under the policy:
```math
V_{k+1}(s) = \sum_{s'}T(s,\pi(s),s')[R(s,\pi(s),s') + \gamma V_k^{\pi}(s') ]
```
2. Compute a policy update


We end up with a linear system that can be solved exactly $\mathcal{O}(n^3)$ or in an iterative, approximate way.

Now we set a fixed policy to start and solve the problem again.
"""

# ╔═╡ 533b30be-0a8b-4339-9643-74dc9247319c
md"""
## Implementation
Below you can find a generic implementation of a MDP.

In the additional file ("PS02\_MDP\_spt.jl"), you can find a generic implentation for dealing with MDP's. The contents of this file can be loaded by using the helper function `ingredients` (hidden in the cell below).
"""

# ╔═╡ 58d4664d-ea90-462a-a68e-0a08262daedc
md"""
Basically the script defines generic implementations for:
* a `MarkovDecisionProcess`
* the `reward` (R)
* the `transition_model` (T)
* the `actions` (a)
Furthermore, it offers implementations for both solution methods discussed in the lectures.
"""

# ╔═╡ 7771740c-9532-4677-88b0-78daeed9c9a7
md"""### Value iteration method
```Julia
function value_iteration(mdp::T; ϵ::Float64=0.001) where {T <: AbstractMarkovDecisionProcess}
    # initialise utility to zero for all states
    U_prime = Dict(collect(Pair(state, 0.0) for state in mdp.states))
    while true
        U = copy(U_prime)
        δ = 0.0
        for state in mdp.states
            U_prime[state] = reward(mdp, state) + 
							mdp.gamma * maximum(map(x -> sum(p*U[newstate] for (p, newstate) in x), 
                                                [transition_model(mdp, state, action) for action in 													actions(mdp, state)]))
            δ = max(δ, abs(U_prime[state] - U[state]));
        end
        if (δ < ((ϵ * (1 - mdp.gamma))/mdp.gamma))
            return U_prime;
        end
    end
end
```
"""

# ╔═╡ 1ac299bd-3b3f-4f49-95c4-f1ec3eacd265
md"""
### Policy iteration method
```Julia
function policy_iteration(mdp::T) where {T <: AbstractMarkovDecisionProcess}
    U = Dict(Pair(state, 0.0) for state in mdp.states)
    Π = Dict(Pair(state, rand(actions(mdp, state))) for state in mdp.states)
    while true
        # compute utility
        U = policy_evaluation(Π, U, mdp)
        unchanged = true
        # update policy
        for state in mdp.states
            A = collect(actions(mdp, state))
            expected_utility = map(x -> sum(p*U[newstate] for (p, newstate) in x),  [transition_model(mdp, state, action) for action in A])
            action = A[argmax(expected_utility)]
            if action != Π[state]
                Π[state] = action
                unchanged = false
            end
        end
        if unchanged
            return Π, U;
        end 
    end
end
```
"""

# ╔═╡ 86cfff4b-30f0-4074-8bd9-99dbac424cfd
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

# ╔═╡ b045f42d-9882-48a2-a1f8-e4c15d2a2e66
MDP = ingredients("./PS02_MDP_spt.jl");nothing

# ╔═╡ a9a6499f-2354-4495-975b-495c2889efba
md"""
## Application - stochastic grid environment
Consider the 4x3 environment from the course. Below you can find implementation for this problem.
"""

# ╔═╡ 5ae0bdfd-c4e6-4f06-8923-9cff67b31ba0
begin

	"""
	    GridMarkovDecisionProcess is a MDP implementation of the grid from chapter 16. 
	
	Obstacles in the environment are represented by `nothing`.
	
	"""
	struct GridMarkovDecisionProcess <: MDP.AbstractMarkovDecisionProcess
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
	function MDP.transition_model(gmdp::GridMarkovDecisionProcess, state::Tuple{Int64, Int64}, action::Nothing)
	    return [(0.0, state)];
	end
	
	"""
	    transition_model
	    
	Return an array of (P(s'|s, a), s') pairs given the state 's' and action 'a'.
	"""
	function MDP.transition_model(gmdp::GridMarkovDecisionProcess, state::Tuple{Int64, Int64}, action::Tuple{Int64, Int64})
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
	nothing

end

# ╔═╡ 88da3995-c366-4e4b-87ea-6d03ab6544eb
md"""
We can define a MDP from the example from the course by defining the initial and terminal states as well as the rewards for each state.
"""

# ╔═╡ 06eb94f5-18b8-420b-bb37-c4bed65672ec
begin
	# environment setup
	initial = (1, 1)
	terminal_states = Set([(2, 4), (3, 4)])
	grid = [-0.04 -0.04 -0.04 -0.04;
			-0.04 nothing -0.04 -1;
			-0.04 -0.04 -0.04 +1]
	# environment creation
	myMDP = GridMarkovDecisionProcess(initial, terminal_states, grid)
end

# ╔═╡ 344b2ed1-c4a5-427d-811d-4a2757c9677d
md"""
If you want to visualize a specific set of actions, you can use the `show_grid` function. If you have a set of actions that leads to a solution and describes what to do for each state, you have a policy for the MDP.
"""

# ╔═╡ b8e0a622-5f7d-4ebc-8758-7907cee24f34
begin
	# Illustration of limited actions (maps: state => action)
	myactions = Dict((1,1) => (0,1), (1,2) => (1,0) )
	# show results
	gridstring = show_grid(myMDP,myactions)
	println("Grid layout and actions:")
	println(gridstring)
end

# ╔═╡ 22070e43-650c-41b4-b4de-050da37f83fe
md"""
We can verify the behavior of our model and the different function for some small examples
"""

# ╔═╡ 7c933a68-a618-4df2-9e73-bff1c3faf4b6
begin
	pos = (2,3)
	action=(1,0)
	# all possible actions are shown
	MDP.actions(myMDP, pos)
end

# ╔═╡ a0ab6856-ce14-4f18-aee1-efb137a89ca8
md"""
Although all actions are identified, invalid postions and action are taken into account in the transition model. You can clearly see this in the example below. You would expect to find (2,2) as a valid case, but as this state does not exist, you get staying in place instead.
"""

# ╔═╡ 3b7213c7-c782-4686-9dc2-a0d522587d3a
tt = MDP.transition_model(myMDP,pos,action)

# ╔═╡ beff15ad-ea24-44a4-912d-5101759ac937
md"""
You can observe something similar when you try an action that does not make sense e.g. trying to quit the space in a corner:
"""

# ╔═╡ f7f02056-6f87-4bbf-b157-a44693b47a6b
let
	state = (1,4)
	[(action, MDP.transition_model(myMDP,state,action)) for action in MDP.actions(myMDP,state)]
end

# ╔═╡ 03712063-cbdf-4680-ae77-994d37970674
md"""
Our transition model works as intended. We can now use value iteration to compute the utilities of the different states.
"""

# ╔═╡ 612b57ce-549e-430c-b040-42aa13830283
VI = MDP.value_iteration(myMDP)

# ╔═╡ 6ce9ef18-2342-4cab-86e7-9e97aeb670f8
md"""
Now that we have the final utility values, we can extract the policy and illustrate it.
"""

# ╔═╡ 8f7ca271-87da-4df5-a80b-ebc59ed121db
pol = MDP.policy_extraction(myMDP, VI)

# ╔═╡ d1734b14-9e9f-4208-9d9b-8be19161f3dc
let
	F = zeros(size(myMDP.grid))
	for (key,val) in VI
		F[key...] = val
	end
	p = heatmap(F,seriescolor=:RdYlGn_11,yflip=true)
	title!("value iteration results (non-terminal reward: -0.04)")
	yticks!(1:3)
	for (key,val) in pol
		dir = text("""$(get(MDP.arrow_characters,val,"?"))""", :center, 28)
		annotate!((key[2],key[1]-1/6, dir))
		annotate!((key[2],key[1]+1/4, """$(round(get(VI,key,0),digits=4))"""))
	end
	p
end

# ╔═╡ 1521e183-e15e-4ac1-ac5b-6a058914bd05
polbis, ubis = MDP.policy_iteration(myMDP)

# ╔═╡ 544c8ce3-0ef6-401e-a0f1-51706004093c
print("Actions to take:\n", join([" $(key) => $(val)" for (key,val) in polbis],"\n"))

# ╔═╡ 40aa0d72-894f-473f-b644-15fcba096aba
let
	F = zeros(size(myMDP.grid))
	for (key,val) in ubis
		F[key...] = val
	end
	p = heatmap(F,seriescolor=:RdYlGn_11,yflip=true)
	title!("policy iteration results (non-terminal reward: -0.04)")
	yticks!(1:3)
	for (key,val) in polbis
		dir = text("""$(get(MDP.arrow_characters,val,"?"))""", :center, 28)
		annotate!((key[2],key[1]-1/6, dir))
		annotate!((key[2],key[1]+1/4, """$(round(get(ubis ,key,0),digits=4))"""))
	end
	p
end

# ╔═╡ b008fbdc-10c3-4f67-b124-9870fc7eaa58
md"""
**Note:**

In the previous example, the rewards of a state depends only on the state itself, i.e. $R(s)$. In practice however, you can encounter a reward that also depends on the action taken, i.e. $R(s,a)$, or even on the action taken and the outcome state, i.e. $R(s,a,s')$.
"""

# ╔═╡ 6dd8d501-ee40-4409-a5b8-5835297fdde0
md"""
## Application - human life cycle
Consider the following human life cycle:
* you can be `tired`, from there you can 
  * work: this will leave you tired
  * workout: this can leave you `tired` or `energetic`
  * sleep: this can leave you `energetic` or `tired`
* you can be `energetic`, from there you can 
  * work: this will leave you `tired` or `energetic`
  * workout: this will leave you `healthier`
* you can be `healthier`, from there you can 
  * work: this will leave you `tired`.


Questions:
1. make a graphical representation of this problem (states, transition model, reward). Suggest some values for the transition probabilities and the rewards.
2. come up with an optimal policy. Do your results make sense?
3. study the influence of the different parameters
4. try to produce a figure similar to 16.7 from the book (i.e. using value iteration, show the evolution of utilities for different states and show the impact of the values of ``\gamma`` on the convergence)
  $(PlutoUI.LocalResource("./img/MPD_param_study.png", :width => 500, :align => "middle"))
5. try to produce a figure similar to 16.8 from the book (i.e. using value iteration, show the evolution of the policy loss and the max error in function of the number of iterations.)
  $(PlutoUI.LocalResource("./img/MPD_param_study_2.png", :width => 500, :align => "middle"))

Notes: you might need to modify the given code in order to make it stop after a number of iterations or write out intermediate results.
"""

# ╔═╡ f00c627d-9226-4f24-82ed-5ed3db86bf3a


# ╔═╡ Cell order:
# ╟─91ed838a-d14f-4a0a-863b-4884e5cfa529
# ╟─61e06d99-e14d-462f-a83a-bffd734408f6
# ╟─763946d5-f8b3-430e-b889-20a2559ed15f
# ╟─25d705d9-c565-418c-933b-5bd6303e7b24
# ╟─22d811c0-5592-45ef-b542-bb7c3d79c812
# ╟─18b033cb-c351-4078-85ec-c7217294a17e
# ╟─eab33b26-56bf-4837-be2a-6b41acf7ab0b
# ╟─46e7fb69-8bd0-4a78-a7df-83e71cde2abc
# ╟─c7c601d8-408e-4227-9c56-0e30f429eb50
# ╟─64530d11-46fd-4450-8593-c0521b9a54a6
# ╟─29226fd6-893e-4ca2-8bfb-6e834a9530bf
# ╟─533b30be-0a8b-4339-9643-74dc9247319c
# ╠═b045f42d-9882-48a2-a1f8-e4c15d2a2e66
# ╟─58d4664d-ea90-462a-a68e-0a08262daedc
# ╟─7771740c-9532-4677-88b0-78daeed9c9a7
# ╟─1ac299bd-3b3f-4f49-95c4-f1ec3eacd265
# ╟─86cfff4b-30f0-4074-8bd9-99dbac424cfd
# ╟─a9a6499f-2354-4495-975b-495c2889efba
# ╠═5ae0bdfd-c4e6-4f06-8923-9cff67b31ba0
# ╟─88da3995-c366-4e4b-87ea-6d03ab6544eb
# ╠═06eb94f5-18b8-420b-bb37-c4bed65672ec
# ╟─344b2ed1-c4a5-427d-811d-4a2757c9677d
# ╠═b8e0a622-5f7d-4ebc-8758-7907cee24f34
# ╟─22070e43-650c-41b4-b4de-050da37f83fe
# ╠═7c933a68-a618-4df2-9e73-bff1c3faf4b6
# ╟─a0ab6856-ce14-4f18-aee1-efb137a89ca8
# ╠═3b7213c7-c782-4686-9dc2-a0d522587d3a
# ╟─beff15ad-ea24-44a4-912d-5101759ac937
# ╠═f7f02056-6f87-4bbf-b157-a44693b47a6b
# ╟─03712063-cbdf-4680-ae77-994d37970674
# ╠═612b57ce-549e-430c-b040-42aa13830283
# ╟─6ce9ef18-2342-4cab-86e7-9e97aeb670f8
# ╠═8f7ca271-87da-4df5-a80b-ebc59ed121db
# ╟─d1734b14-9e9f-4208-9d9b-8be19161f3dc
# ╠═1521e183-e15e-4ac1-ac5b-6a058914bd05
# ╠═544c8ce3-0ef6-401e-a0f1-51706004093c
# ╟─40aa0d72-894f-473f-b644-15fcba096aba
# ╟─b008fbdc-10c3-4f67-b124-9870fc7eaa58
# ╟─6dd8d501-ee40-4409-a5b8-5835297fdde0
# ╠═f00c627d-9226-4f24-82ed-5ed3db86bf3a
