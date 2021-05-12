### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ 61e06d99-e14d-462f-a83a-bffd734408f6
using Plots

# ╔═╡ 2b1cdf75-d93d-413a-b3f7-37eb9ea54910
md"""
In the additional file ("PS14\_MDP\_spt.jl"), you can find a generic implentation for dealing with MDP's. 
"""

# ╔═╡ ee9e8098-5bb9-4598-bc7d-095412b8cd73
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
end

# ╔═╡ 9c9e40ce-6ed6-4dd5-951e-074e6bd3a2da
MDP = ingredients("./PS14_MDP_spt.jl")

# ╔═╡ eac907a6-b230-11eb-04f1-5df27e70da8b
md"""
# Complex decisions
A sequential decision problem for a fully observable, stochastic environment with a Markovian transition model and additive rewards is called a Markov decision process (MDP). It consists of the following items:
* a set of states (with its initial state $s_0$)
* a set of actions $\text{ACTION}(s)$ of actions in each state
* a transition model $P(s'|s,a)$
* a reward function $R(s)$

In MDP chance node represent uncertainty about what might happen based on (state,action). A Q-state (s,a) is when you were in a state and took an action.

"""

# ╔═╡ 981096f5-6948-4355-93b2-0fcf453213a5
md"""
## Transforming a problem into an MDP
Below we have a set of problem that we want to transform into the MDP formalism. Think about the states, transition probabilties and rewards. Provide the components of the MDP formalism in a graphical way.
1. a cleaning robot needs to collect trash. The robot has limited autonomy. Possible activities include include searching for trash, waiting  or recharging itself.
2. a human wants to maximize its earnings. The human can be tired, energetic or healthier. in function of how the human is feeling, he can sleep, work (=earn money) or do a workout. Working always fatigues the humen. Doing a workout makes him feel more energetic when tired and makes him healthier when energetic.
3. consider the example of the lectures with the robot in the 4x3 grid. provide the MDP formalism.

How would you keep track of the states and their transition probabilities in practice?
"""

# ╔═╡ 46e7fb69-8bd0-4a78-a7df-83e71cde2abc
md"""
## Solving an MDP
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
	myMDP = MDP.GridMarkovDecisionProcess(initial, terminal_states, grid)
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
	gridstring = MDP.show_grid(myMDP,myactions)
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
Altough all actions are identified, invalid postions and action are taken into account in the transition model. You can clearly see this in the example below. You would expect to find (2,2) as a valid case, but as this state does not exist, you get staying in place instead.
"""

# ╔═╡ 3b7213c7-c782-4686-9dc2-a0d522587d3a
tt = MDP.transition_model(myMDP,pos,action)

# ╔═╡ beff15ad-ea24-44a4-912d-5101759ac937
md"""
You can observe something similar when you try an action that does not make sense e.g. trying to quit the space in a corner:
"""

# ╔═╡ f7f02056-6f87-4bbf-b157-a44693b47a6b
let
	state = (4,4)
	[MDP.transition_model(myMDP,state,action) for action in MDP.actions(myMDP,state)]
end

# ╔═╡ 03712063-cbdf-4680-ae77-994d37970674
md"""
Our transition model works as intended.
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
This algorithm will converge to the unique optimal values after a number of iterations after being initialized to zero.
"""

# ╔═╡ 612b57ce-549e-430c-b040-42aa13830283
VI = MDP.value_iteration(myMDP)

# ╔═╡ 6ce9ef18-2342-4cab-86e7-9e97aeb670f8
md"""
Now that we have the final values, we can illustrate it
"""

# ╔═╡ 6e90ab1e-dd41-4228-9e01-d3bfada8c387
md"""
#### Obtaining a policy from value iteration
Once you have the solutions of the problem, you can obtain the optimal policy by doing policy extraction:
```math
\pi^{*}(s)=  \underset{a}{\text{argmax}} \sum_{s'}T(s,a,s')[R(s,a,s') + \gamma V^{*}(s,a,s')]
```
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
	title!("value iteration results")
	yticks!(1:3)
	for (key,val) in pol
		dir = text("""$(get(MDP.arrow_characters,val,"?"))""", :center, 28)
		annotate!((key[2],key[1]-1/6, dir))
		annotate!((key[2],key[1]+1/4, """$(round(get(VI,key,0),digits=4))"""))
	end
	p
end

# ╔═╡ 29226fd6-893e-4ca2-8bfb-6e834a9530bf
md"""
### Policy iteration
In the previous section we have used the non-linear $\max$ operator. If you impose a policy on the system, you no longer have to do this and this reduces the computation from $\mathcal{O}(S^2A)$ to $\mathcal{O}(S^2)$. The update is given by

```math
V_{k+1}(s) = \sum_{s'}T(s,\pi(s),s')[R(s,\pi(s),s') + \gamma V_k^{\pi}(s') ]
```
We end up with a linear system that can be solved exactly $\mathcal{O}(n^3)$ or in an iterative, approximate way.

Now we set a fixed policy to start and solve the problem again.
"""

# ╔═╡ 1521e183-e15e-4ac1-ac5b-6a058914bd05
polbis,ubis = MDP.policy_iteration(myMDP)

# ╔═╡ 40aa0d72-894f-473f-b644-15fcba096aba
let
	F = zeros(size(myMDP.grid))
	for (key,val) in ubis
		F[key...] = val
	end
	p = heatmap(F,seriescolor=:RdYlGn_11,yflip=true)
	title!("policy iteration results")
	yticks!(1:3)
	for (key,val) in polbis
		dir = text("""$(get(MDP.arrow_characters,val,"?"))""", :center, 28)
		annotate!((key[2],key[1]-1/6, dir))
		annotate!((key[2],key[1]+1/4, """$(round(get(VI,key,0),digits=4))"""))
	end
	p
end

# ╔═╡ 3f1d3609-d3c7-42cc-bd19-1a18a1df2d8c
md"""
# TO DO:
* complete docstrings of functions
* add additional examples/tasks
"""

# ╔═╡ Cell order:
# ╠═61e06d99-e14d-462f-a83a-bffd734408f6
# ╟─2b1cdf75-d93d-413a-b3f7-37eb9ea54910
# ╟─ee9e8098-5bb9-4598-bc7d-095412b8cd73
# ╠═9c9e40ce-6ed6-4dd5-951e-074e6bd3a2da
# ╟─eac907a6-b230-11eb-04f1-5df27e70da8b
# ╟─981096f5-6948-4355-93b2-0fcf453213a5
# ╟─46e7fb69-8bd0-4a78-a7df-83e71cde2abc
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
# ╟─c7c601d8-408e-4227-9c56-0e30f429eb50
# ╠═612b57ce-549e-430c-b040-42aa13830283
# ╟─6ce9ef18-2342-4cab-86e7-9e97aeb670f8
# ╟─6e90ab1e-dd41-4228-9e01-d3bfada8c387
# ╠═8f7ca271-87da-4df5-a80b-ebc59ed121db
# ╠═d1734b14-9e9f-4208-9d9b-8be19161f3dc
# ╟─29226fd6-893e-4ca2-8bfb-6e834a9530bf
# ╠═1521e183-e15e-4ac1-ac5b-6a058914bd05
# ╠═40aa0d72-894f-473f-b644-15fcba096aba
# ╟─3f1d3609-d3c7-42cc-bd19-1a18a1df2d8c
