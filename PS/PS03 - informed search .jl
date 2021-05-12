### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ ea9468a2-710f-11eb-2284-6d8af448d9eb
# dependencies
begin
	using PlutoUI
	using DataStructures
	using Logging
	Logging.global_logger(Logging.SimpleLogger(stdout, LogLevel(-5000)  ) )
	Logging.disable_logging(LogLevel(0))
end

# ╔═╡ 49d11f58-713c-11eb-20bb-e3a3dc889b6b
begin
	using LightGraphs, SimpleWeightedGraphs, Plots, LinearAlgebra

	"""
		mst(A::Array{<:Number, 2})

	From an adjacency matrix return a dictionary of possible 
	destinations (including weights, returned a  tuple)
	"""
	function mst(A::Array{<:Number, 2})
		G = SimpleWeightedGraph(A) # initialise graph
		MST = kruskal_mst(G)       # calculate MST
		# translate MST to dict
		d = Dict{Int64, Array{Tuple{Int64, Float64},1}}()
		for edge in MST
		   push!(get!(d, edge.src, Array{Tuple{Int64, Float64},1}()), (edge.dst, edge.weight))
		end
		return d
	end
end;

# ╔═╡ 7782b3b2-710f-11eb-1a92-e12b8b1f3299
md"""
# A-star and the eight puzzle

Consider the eight puzzle, i.e. a grid of nine cases, with 8 integer values in them. The goal is to start with the empty square and then have all numbers in increasing order.

|initial   |       | state  |
|---|---|---|
| 7 | 2 | 4 |
| 5 |   | 6 |
| 8 | 3 | 1 |

|goal   |       | state  |
|---|---|---|
|   | 1 | 2 |
| 3 | 4 | 5 |
| 6 | 7 | 8 |

The objective of this session is to implement different heuristics and compare the performance of the A-star algorithm with the tree search methods implemented beforehand.

Below you find a number of generic and specific functions that allow you to identify the possible actions for a given state and to determine the next state from a previous one. For didactic purposes, there also is a function that allows you to visualize the current state.
"""

# ╔═╡ 58b8349c-7110-11eb-1aab-f7c716a12c57
md"""## Generic functions"""

# ╔═╡ 13d314dc-7110-11eb-3475-2126f5d414b7
begin
	"""
	Type used for representation of a node in the search tree.

	The node type will be inherited from the type of the state that is passed as an argument to the inner constructor

	Fields:
		- state: a node needs a state (of non-specified type T)
		- parent: each node has either a parent or nothing (in case of the root node)
		- action: the action that was taken to get from the previous node to the current one
		- pathcost: total cost of the current path (each contribution should be non-negative)
		- depth: to illustrate the depth of the tree and to be able to implement depth limited search.
	"""
	struct Node{T}
		state::T
		parent::Union{Nothing, Node, UInt}
		action::Union{Nothing, String, Int64, Tuple, UnitRange}
		pathcost::Float64
		depth::Int
		function Node(state::T, parent::Union{Nothing, Node, UInt}=nothing, action::Union{Nothing, String, Int64, Tuple, UnitRange}=nothing, pathcost=0.0, depth::Int=0) where {T}
			return new{T}(state, parent, action, pathcost, depth)
		end
	end

	"""
	Type used for representation of a problem with an initial and a goal state of type T
	"""
	struct Problem{T}
		initstate::T
		goalstate::T
		function Problem(initstate::T, goalstate::T) where {T}
			return new{T}(initstate, goalstate)
		end
	end

	"""
		path(n::Node)

	Return sequence of nodes to come to a solution (inluding the root node)
	"""
	function path(n::Node)
		track = Array{Node,1}()
		node = n
		pushfirst!(track, node)
		while node.parent ≠ nothing
			node = node.parent
			pushfirst!(track, node)
		end
		return track
	end

	"""
		solution(x::Array{Node,1})

	Return sequence of actions to come to a solution
	"""
	function solution(x::Array{Node,1})
		return [node.action for node in x[2:end]]
	end
end;

# ╔═╡ 423469c0-7110-11eb-0cef-03540836525b
md"""## Problem specific functions"""

# ╔═╡ 68def13a-7110-11eb-29cf-8da0df67b178
begin
	
	"""
		goaltest_puzzle(p::Problem, n::Node)

	Compare problem's endstate with current state of a node
	"""
	function goaltest_puzzle(p::Problem, n::Node)
		if p.goalstate == n.state
			return true
		else
			return false
		end
	end
	
	"""
		findactions_puzzle(state::NTuple{N,Int64}) where {N}

	Find all possible outcomes of actions, independent of gridsize
	"""
	function findactions_puzzle(state::NTuple{N,Int64}) where {N}
		d = Int(sqrt(N))
		# all possible actions (only four exist to move the blank case):
		possible_actions = ["UP","DOWN","LEFT","RIGHT"]
		# find location of empty case
		ind = findfirst(x->x==0, state)
		# remove options
		if ind <= d 
			filter!(x -> x ≠ "UP", possible_actions)
		end
		if ind > N - d
			filter!(x -> x ≠ "DOWN", possible_actions)
		end
		if ind % d  == 1
			filter!(x -> x ≠ "LEFT", possible_actions)
		end
		if ind % d  == 0
			filter!(x -> x ≠ "RIGHT", possible_actions)
		end

		return possible_actions
	end


	"""
		applyaction_puzzle(state::NTuple{N,Int64}, action::String) where {N}

	Go from one state to another by applying an action.
	"""
	function applyaction_puzzle(state::NTuple{N,Int64}, action::String) where {N}
		d = Int(sqrt(N))
		# make copy of state
		state = collect(state)
		# find location of empty case
		ind = findfirst(x->x==0, state)
		# operations and index shifts   ## Maybe make this a constant for speed gains
		ops = Dict("UP"=>-d, "DOWN"=>d, "LEFT"=>-1, "RIGHT"=>1)
		state[ind], state[ind + ops[action]] = state[ind + ops[action]], state[ind]
		return Tuple(state)
	end


	"""
		prettyprint(state::Array{Int,1})

	Supporting function to visualize the playing field
	"""
	function prettyprint(state::NTuple{N,Int64}) where {N}
		d = Int(sqrt(N))
		function morph(x::Int)
			if x == 0
				return "  "
			elseif x < 10
				return " $(x)"
			elseif x >= 10
				return "$(x)"
			end
		end
		msg = "Game layout:\n\n"
		for i = 1:d
			msg = msg *"| " *  prod(["$(morph(state[(i-1)*d+j])) | " for j in 1:d]) * "\n"
		end
		println(msg)
	end
end;

# ╔═╡ b5309106-7110-11eb-3631-c1484cc8a5d9
md"""## Illustration of a small game"""

# ╔═╡ c28473ea-7110-11eb-2129-cdc0aab5444e
begin
	println("Illustration of an easy case (2 steps):")
	startstate = Tuple([3, 1, 2, 4, 0, 5, 6, 7, 8])
	endstate = Tuple(collect(0:8))
	p = Problem(startstate, endstate)
	prettyprint(p.initstate)
	actions = ["LEFT","UP"]
	state = p.initstate
	for action in actions
		state = applyaction_puzzle(state, action)
		prettyprint(state)
	end
end

# ╔═╡ ccfd8924-7110-11eb-3890-6d9d80cfb96b
md"""## Astar algorithm 
Below you find the treesearch Astar algorithm implementation. Analyze the way it works ."""

# ╔═╡ d629933a-7110-11eb-39ad-7582f1db3d62
begin
	"""
		treesearchAstar(p::Problem; findactions::Function, applyaction::Function, goaltest::Function, g::Function, h::Function, kwargs...)

	Relatively generic implementation of A* tree search.
	Determines sequence of nodes and transformations.

	keyword arguments:
		- g(p, node, action)::Function: calculate the path cost from one node to the next one
		- h(state, goalstate)::Function: calculate the heuristic for a current state towards a goal state

		- to allow for extra output during the tree traversal, you need to enable a logger that can actually log this. e.g. 
						Logging.global_logger(Logging.SimpleLogger(stdout, LogLevel(-5000)  ) )
						Logging.disable_logging(LogLevel(-5000))
	"""
	function treesearchAstar(p::Problem;
						 	findactions::Function,
							applyaction::Function,
							goaltest::Function,
							g::Function, 
							h::Function, kwargs...)
		@debug "\n" * "#"^29 * "\n# Astar tree search # \n" * "#"^29
		node = Node(p.initstate)                                             # initialise root node
		fringe = DataStructures.PriorityQueue{Node, typeof(node.pathcost)}() # create Priority Queue
		enqueue!(fringe, node, node.pathcost + h(p, node; action=nothing, kwargs...))   # place root node in the Queue
		@debug "Current queue: $(fringe)"
		while !isempty(fringe)                                               # expand the fringe
			node = dequeue!(fringe)                                          # pop element from queue
			if goaltest(p, node; kwargs...)
				@debug "Solution found!"
				return path(node), solution(path(node))
			end
			@debug "Current node: $(node), current queue: $(fringe)"
			@debug "Possible action: $(findactions(node.state))"
			for action in findactions(node.state; kwargs...)
				child = Node(applyaction(node.state, action; kwargs...), node, action, g(p, node, action; kwargs...), node.depth + 1)
				@debug "new child: $(child)"
				enqueue!(fringe, child, child.pathcost + h(p, node; action=action, kwargs...))
				#enqueue!(fringe, child, child.pathcost + h(node.state, p.goalstate; kwargs...))
			end
		end
		@warn "failed to find a solution"
		return nothing
	end
end;

# ╔═╡ 902a810e-7111-11eb-1c4f-ef75bbcac693
md"""
* Implement both heuristics from the course (number of mismatches & manhattan mismatches)
* Compare the solution with previous methods (computation time + memory cost) on the case below:

```julia
    # Illustrations of a harder case:
    startstate = Tuple([1, 2, 3, 4, 0, 5, 6, 7, 8])
    endstate = Tuple(collect(0:8))
    p = Problem(startstate, endstate)
```

"""

# ╔═╡ 85073ccc-75e7-11eb-0d0f-c1c178e4e57e


# ╔═╡ 07f44e1c-713b-11eb-25b0-b9824cc5d251
md"""
## The Astar graph search method
Additional gains can be obtained by avoiding to visit the same state again (do you remember the requirements for the heuristic function in order for this to lead to the best solution?). Take the Astar algorithm from before and transform the tree search to graph search. One again, compare this to the previous methods.
"""

# ╔═╡ 369047da-713b-11eb-3f47-29e522f17fb5
begin
	"""
		Astargraphsearch(p::Problem; g::Function, h::Function; kwargs...)

	Relatively generic implementation of A* tree search.
	Determines sequence of nodes and transformations.

	keyword arguments:
		- g(p, node, action)::Function: calculate the path cost from one node to the next one
		- h(state, goalstate)::Function: calculate the heuristic for a current state towards a goal stat

		- to allow for extra output during the tree traversal, you need to enable a logger that can actually log this. e.g. 
						Logging.global_logger(Logging.SimpleLogger(stdout, LogLevel(-5000)  ) )
						Logging.disable_logging(LogLevel(-5000))
	"""
	function Astargraphsearch(p::Problem; 
							findactions::Function,
							applyaction::Function,
							goaltest::Function,
							g::Function, 
							h::Function, kwargs...)
		@debug "\n" * "#"^29 * "\n# Astar tree search # \n" * "#"^29
		# initialise root node
		node = Node(p.initstate)       
		# create Priority Queue
		fringe = DataStructures.PriorityQueue{Node, typeof(node.pathcost)}() 
		explored = Set{typeof(p.initstate)}()
		# place root node in the Queue
		enqueue!(fringe, node, node.pathcost + h(p, node; action=nothing, kwargs...))   
		@debug "Current queue: $(fringe)"
		# expand the fringe
		while !isempty(fringe) 
			# pop element from queue
			node = dequeue!(fringe)                                          
			if goaltest(p, node; kwargs...)
				@debug "Solution found!"
				return path(node), solution(path(node))
			end
			push!(explored, node.state)
			@debug "Current node: $(node), current queue: $(fringe)"
			@debug "Possible action: $(findactions(node.state))"
			for action in findactions(node.state)
				child = Node(applyaction(node.state, action), node, action, g(p, node, action; kwargs...), node.depth + 1)
				@debug "new child: $(child)"
				if child.state ∉ explored
					enqueue!(fringe, child, child.pathcost + h(p, node; action=action, kwargs...))
				end
			end
		end
		@warn "failed to find a solution"
		return nothing
	end

end;

# ╔═╡ 9b3d2832-75e7-11eb-0372-35f7b9ed402b


# ╔═╡ b35e5664-7143-11eb-1e75-17b00177d1f9
md"""
## Additional problems
We have seen several search algorithms, each time illustrated by one or more examples. The difficulty of this kind of problems is not so much in the implementation of an algorithm, but in formalising a problem and choosing a correct heuristic. The following problems will illustrate this.
"""

# ╔═╡ a653d9a8-7143-11eb-3812-396c3ec732bd


md"""
### 2D grid

$(PlutoUI.LocalResource("./img/fig3-9.png", :width => 500, :align => "middle"))
Consider the unbounded version of the regular 2D grid shown in Figure 3.9. The start state is at the origin, (0,0), and the goal state is at (x, y).
1. What is the branching factor b in this state space?
2. How many distinct states are there at depth k (for k > 0)?
3. What is the maximum number of nodes expanded by breadth-first tree search?
4. Is h = |u − x| + |v − y| an admissible heuristic for a state at (u, v)? Explain.
"""

# ╔═╡ 28f0121c-713c-11eb-2aff-e7a9c592fa53
md"""

### TSP
The [traveling salesperson problem (TSP)](https://en.wikipedia.org/wiki/Travelling_salesman_problem) can be solved with the [minimum-spanning-tree (MST)](https://en.wikipedia.org/wiki/Minimum_spanning_tree) heuristic, which estimates the cost of completing a tour, given that a partial tour has already been constructed. The MST cost of a set of cities is the smallest sum of the link costs of any tree that connects all the cities.

1. Show how this heuristic can be derived from a relaxed version of the TSP. What relaxation has been applied?
2. Think about how you can implement this problem in the general framework (what is a state, how do you check if a goalstate is realised, how do you define a transition and possible candidates...)
2. Show that the MST heuristic dominates straight-line distance. Compare the following heuristics:
    - MST
    - straight-line distance
    - nearest neighbor
3. Write a problem generator for instances of the TSP where cities are represented by random points in the unit square.
4. Compare the Astar graph search with what you have learned previously (e.g. solve it as a LP problem) 

Below you can find a small example of the MST calculation for a tiny network that can be used for further three search.
"""

# ╔═╡ 99bbb244-713c-11eb-34be-d78f80834d46
# small illustration on a distance matrix
begin
	A = [0 30 5 7 0;
		30 0  1  0 0;
		 5 1  0  10 10;
		7 0  10  0 0;
		0 0 10 0 0]
	mst(A)
end

# ╔═╡ 8b9fb7a0-7146-11eb-089c-f5c25325324b
begin
	"""
		goaltest_TSP(p::Problem, n::Node; kwargs...)

	Compare problem's endstate with current state of a node.
	 (slightly modified the goaltest function to check all visited)
	"""
	function goaltest_TSP(p::Problem, n::Node; kwargs...)
		if p.goalstate[2] == union(n.state[2], n.state[1])
			return true
		else
			return false
		end
	end
	
	
	"""
		solution_TSP(x::Array{Node,1})

	Return sequence of action to come to a solution
	 (slightly modified solution so it returns a closed loop)
	"""
	function solution_TSP(x::Array{Node,1})
		return [x[1].state[1];[node.action for node in x[2:end]]...;x[1].state[1]]
	end
	
	"""
		findactions_TSP(state::Tuple{Int64,Set{Int64}})

	Find all possible actions for the travelling salesman
	i.e. exlude all visited locations and the current location
	"""
	function findactions_TSP(state::Tuple{Int64,Set{Int64},Int64}; kwargs...)
		return setdiff(Set(1:state[3]), state[2], state[1])
	end
	
	"""
		applyaction_TSP(state::Tuple{Int64,Set{Int64}}, action::Int64)

	Go from one state to another by applying an action.
	"""
	function applyaction_TSP(state::Tuple{Int64,Set{Int64},Int64}, action::Int64; kwargs...)
		# new location
		newpos = action
		# update visited list
		newvisited = copy(state[2])
		push!(newvisited, state[1])

		return (newpos, newvisited, state[3])
	end
	
	""" 
		g_TSP(p::Problem, node::Node, action::Int; kwargs...)

	Path cost starting from a node and taking a specific action

	Required kwarg: "wm" = weight matrix describing the distance between each node in the global graph

	"""
	function g_TSP(p::Problem, node::Node, action::Int; kwargs...)
		cost = node.pathcost 
		extra = kwargs[:wm][node.state[1], action]
		#@warn kwargs[:wm]
		#@warn "node depth: $(node.depth), cost: $(node.pathcost), extra: $(extra)"
		return cost + extra
	end
	
	
	""" 
		h_TSP(p::Problem, node::Node; action::Int64, kwargs...)

	MST heuristic, which estimates the cost of completing the tour, 
	given that a partial tour has already been constructed. 

	Required kwarg: "wm" = weight matrix describing the distance between each node in the global graph

	"""
	function h_TSP(p::Problem, node::Node; action::Union{Nothing,Int64}, kwargs...)
		# in case of nothing as action
		if isnothing(action)
			return 0
		end
	
		current = node.state[1]
		vis = node.state[2]
		vcount = node.state[3]
		# subgraph weight matrix: 
		# remove visited nodes, current state or action from global graph
		inds = [x for x in 1:vcount if x∉[current, action, vis...]]
		H_weights = kwargs[:wm][inds,inds]
		H = SimpleWeightedGraph(H_weights)   
		try
			# calculate MST
			MST = prim_mst(H) 
			# return estimate of remaining tour length
			return sum([weights(H)[e.src, e.dst] for e in MST])
		catch
			return 0
		end
	end
end;

# ╔═╡ c6fda06e-713c-11eb-3587-47c7faed0d07
begin
	# visualisations

	"""
		plotgraph(g::SimpleWeightedGraph, d::Dict,draw_MST::Bool=true,wf::Real=5)

	Visualize the network with its edges and their weights

	parameters:
		- g : SimpleWeightedGraph
		- d : Dict
			dictionary that links a node to a coordinate pair
		- draw_MST : Bool
			if true, also draws the MST on the figure
		- wf : Real
			multiplication factor for the edge weigth to obtain a nicer visualisation
	"""
	function plotgraph(g::SimpleWeightedGraph, d::Dict;wf::Real=5,draw_MST::Bool=true, draw_weights::Bool=false)
		p = plot(legend=:bottomright)
		for i in 1:length(d)
			annotate!([d[i][1]], [d[i][2]] .+0.05, "$(i)")
		end
		for e in edges(g)
			plot!([d[e.src][1], d[e.dst][1]],[d[e.src][2], d[e.dst][2]], 
				markershape=:circle, linewidth=e.weight*wf, color=:black, markeralpha = 0.5,
				linealpha=0.2, label="", markercolor=:black, markersize=5)
			if draw_weights
				annotate!([(d[e.src][1] + d[e.dst][1])/2], [(d[e.src][2] + d[e.dst][2])/2],"$(e.weight)", color=:blue);
			end
		end
		plot!([],[],markershape=:circle, color=:black,
				linealpha=0.2, label="Full network", markercolor=:black, markersize=5, markeralpha = 0.5)
		if draw_MST
			MST = prim_mst(g, weights(g))
			for e in MST
				plot!([d[e.src][1], d[e.dst][1]],[d[e.src][2], d[e.dst][2]], 
						markershape=:circle, linewidth=weights(g)[e.src,e.dst]*wf, color=:red,
						linealpha=0.8, label="", linestyle=:solid, markeralpha = 0)

			end
			plot!([],[],linealpha=0.8, label="MST", color=:red, linestyle=:solid)
		end
		xlims!(0,1)
		ylims!(0,1)
		plot!(legend=:best)
		return p
	end

	"""
		plotsol(fp::Array{Int64,1}, nodedict::Dict)

	Function that plots the final TSP solution NOK!
	"""
	function plotsol(fp::Array{Int64,1}, nodedict::Dict)
		#settings = Dict(:arrow => :arrow, :lw => 2, :color=>:blue, :label=>"")
		p = plot()
		startnode = pop!(filter(!in(fp),keys(nodedict)))
		plot!([nodedict[startnode][1]; nodedict[fp[1]][1]],
			  [nodedict[startnode][2]; nodedict[fp[1]][2]],
			  arrow=:arrow, color=:blue,label="")
		for i in 1:length(fp)-1
			x = [nodedict[fp[i]][1]; nodedict[fp[i+1]][1]]
			y = [nodedict[fp[i]][2]; nodedict[fp[i+1]][2]]
			plot!(x,y, arrow=:arrow, color=:blue,label="")
		end
		plot!([nodedict[fp[end]][1]; nodedict[startnode][1]],
			  [nodedict[fp[end]][2]; nodedict[startnode][2]],
			  arrow=:arrow, color=:blue,label="")
		xlims!(0,1)
		ylims!(0,1)
		return p
	end
	
end;

# ╔═╡ c6903b96-713c-11eb-3954-c7fda49d3e35
begin 
	# Play and test case
	N = [(0.5, 0.9);(0.1,0.1); (0.9,0.1); (0.5,0.5)]
	nodedict = Dict( i => N[i] for i in 1:length(N)) # nodes dict
	dm = Float64[0 10 15 20;
		 10 0 35 25;
		 15 35 0 30;
		 20 25 30 0]
	G = SimpleWeightedGraph(dm)
	
	initstate = (1, Set(Array{Int64,1}()),4)
	goalstate = (1, Set(collect(1:length(N))),4)
	P = Problem(initstate, goalstate)


	let
		println("Astar graph search with for TSP with MST heuristic")
		@time (res, actions) = Astargraphsearch(P, findactions=findactions_TSP, 													applyaction=applyaction_TSP, 		
												goaltest=goaltest_TSP, 
												g=g_TSP, h=h_TSP, wm=dm);
		

		plot(plotgraph(G,nodedict, wf=0.51, draw_weights=true), plotsol(actions, nodedict), size=(600,300))
		title!("TSP solution with graph\nsearch and MST heuristic")
end

end

# ╔═╡ 39578d8a-7167-11eb-0e98-314089370666
begin
	# specific generating functions
	"""
		problemgenerator(n::Int)

	generate n points in the unit square
	"""
	function problemgenerator(n::Int)
		return [(rand(), rand()) for _ in 1:n]
	end

	"""
		distancematrix(x::Array{Tuple{Float64,Float64},1}, p::Real=1)

	Generate distancematrix between all points.

	x is an array [n x 1] and p is the L_p-norm you want to use for the distancematrix
	"""
	function distancematrix(x::Array{Tuple{Float64,Float64},1}, p::Real=1)
		n = length(x)
		A = zeros(n,n)
		for (i,j) in collect(Base.Iterators.product(1:n, 1:n))
			A[i,j] = norm(x[i] .- x[j], p)
		end
		return A
	end
end;

# ╔═╡ 4cea4e00-7167-11eb-17de-abf6c558f9a9
let
	n = 15 # number of nodes
	N = problemgenerator(n) # nodes
	nodedict = Dict( i => N[i] for i in 1:length(N)) # nodes dict
	A = distancematrix(N,1) # distance matrix using manhattan norm
	G = SimpleWeightedGraph(A) # graph
	
	initstate = (1, Set(Array{Int64,1}()),n)
	goalstate = (1, Set(collect(1:length(N))),n)
	P = Problem(initstate, goalstate) 
	
	println("Astar graph search with for TSP (n = $(n)) with MST heuristic")
		@time (res, actions) = Astargraphsearch(P, findactions=findactions_TSP, 													applyaction=applyaction_TSP, 		
												goaltest=goaltest_TSP, 
												g=g_TSP, h=h_TSP, wm=A);
	
	
	plot(plotgraph(G,nodedict, wf=0.51), plotsol(actions, nodedict), size=(600,300))
	title!("TSP solution with graph\nsearch and MST heuristic")
end

# ╔═╡ Cell order:
# ╠═ea9468a2-710f-11eb-2284-6d8af448d9eb
# ╟─7782b3b2-710f-11eb-1a92-e12b8b1f3299
# ╟─58b8349c-7110-11eb-1aab-f7c716a12c57
# ╠═13d314dc-7110-11eb-3475-2126f5d414b7
# ╟─423469c0-7110-11eb-0cef-03540836525b
# ╠═68def13a-7110-11eb-29cf-8da0df67b178
# ╠═b5309106-7110-11eb-3631-c1484cc8a5d9
# ╠═c28473ea-7110-11eb-2129-cdc0aab5444e
# ╟─ccfd8924-7110-11eb-3890-6d9d80cfb96b
# ╠═d629933a-7110-11eb-39ad-7582f1db3d62
# ╟─902a810e-7111-11eb-1c4f-ef75bbcac693
# ╠═85073ccc-75e7-11eb-0d0f-c1c178e4e57e
# ╟─07f44e1c-713b-11eb-25b0-b9824cc5d251
# ╠═369047da-713b-11eb-3f47-29e522f17fb5
# ╠═9b3d2832-75e7-11eb-0372-35f7b9ed402b
# ╟─b35e5664-7143-11eb-1e75-17b00177d1f9
# ╟─a653d9a8-7143-11eb-3812-396c3ec732bd
# ╟─28f0121c-713c-11eb-2aff-e7a9c592fa53
# ╠═49d11f58-713c-11eb-20bb-e3a3dc889b6b
# ╠═99bbb244-713c-11eb-34be-d78f80834d46
# ╠═8b9fb7a0-7146-11eb-089c-f5c25325324b
# ╠═c6fda06e-713c-11eb-3587-47c7faed0d07
# ╠═c6903b96-713c-11eb-3954-c7fda49d3e35
# ╠═39578d8a-7167-11eb-0e98-314089370666
# ╠═4cea4e00-7167-11eb-17de-abf6c558f9a9
