### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ ea9468a2-710f-11eb-2284-6d8af448d9eb
# dependencies
begin
	using Pkg
	Pkg.activate(joinpath(@__DIR__,".."))
	
	using PlutoUI
	using DataStructures
	using Logging
	Logging.global_logger(Logging.SimpleLogger(stdout, LogLevel(-5000)  ) )
	using Graphs, SimpleWeightedGraphs, Plots, LinearAlgebra # for the TSP part
	TableOfContents(title="Search")
end

# ╔═╡ 794171c4-ba1a-4d75-a3fa-b250d36e8041
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

# ╔═╡ 4d7293cb-1646-46c4-a324-e33f32915d85
Logging.disable_logging(LogLevel(-5000))

# ╔═╡ 5690f528-fdbc-40c6-9a38-9fa20a36fb75
md"""
# Informed search
For informed search, one uses domain-specific knowledge obout the location of the goals. This knowledge appears in the form of a heuristic function ``h(n)``. The heuristic function represents the estimated cost of the cheapest path from the state at node `n` to a goal state.

### Quick questions
* Is ``A^{*}`` complete?
* is ``A^{*}`` cost-optimal?

### True or false? Argument or provide a counterexample
* Depth-first search always expands at least as many nodes as A∗ search with an admissible heuristic.
* ``h(n) = 0`` is an admissible heuristic for the 8-puzzle.
* ``A^{∗}`` is of no use in robotics because percepts, states, and actions are continuous. 
* Breadth-first search is complete even if zero step costs are allowed.
* Assume that a rook can move on a chessboard any number of squares in a straight line, vertically or horizontally, but cannot jump over other pieces. Manhattan distance is an admissible heuristic for the problem of moving the rook from square A to square B in the smallest number of moves.

### Application: 2D grid

$(PlutoUI.LocalResource("./img/fig3-9.png", :width => 500, :align => "middle"))
Consider the unbounded version of the regular 2D grid shown in Figure 3.9. The start state is at the origin, (0,0), and the goal state is at (x, y).
1. What is the branching factor b in this state space?
2. How many distinct states are there at depth k (for k > 0)?
3. What is the maximum number of nodes expanded by breadth-first tree search?
4. Is h = |u − x| + |v − y| an admissible heuristic for a state at (u, v)?
5. How many nodes are expanded by A-star graph search using h?
6. Is h still admissible if some links are removed from the grid?
7. Is h still admissible if links to non-adjacent states are added to the grid?


## ``A^{*}`` search
The ``A^{*}`` search algorithm is a best-first search that uses an evaluation function which combines the path cost from the initial node to node n en the estimate of the shortest path from n to a goal state
```math
f(n)=g(n)+h(n)
```

### Implementation
Below you find an implementation for both tree and graph search using A-star. 
* Analyze the way they work, what is the difference between both? 
* Use ``A^{*}`` for the 2D grid to confirm the the ideas from the previous exercise.
"""

# ╔═╡ 49f2c47b-ae96-42bd-ac84-ebd63d410799
begin
	abstract type Problem end
	abstract type Node end
end

# ╔═╡ 4063607a-ea41-4ccc-90c0-5c729e620572
begin
	
	"""
		Astartreesearch(p::Problem; 
						findactions::Function, 
						applyaction::Function, 
						goaltest::Function, 
						g::Function, 
						h::Function, 
						solution::Function,
						kwargs...)

	Generic implementation of A* graph search. Determines sequence of actions.

	## keyword arguments:
		- findactions::Function: get available actions from a state 
		- applyaction::Function: get new node from applying an action on a state
		- goaltest::Function: evaluate if a node is in a goalstate
		- g(p, node, newstate)::Function: calculate the path cost from one node to the next state
		- h(p, node)::Function: calculate the heuristic for a current state towards a goal state
		- solution(node)::Function: trace the set of actions required to end up at the solution
		- kwargs: additions keyword argument pairs

	## Notes:
	to allow for extra output during the tree traversal, you need to enable a logger that can actually log the debug level.
	"""
	function Astartreesearch(p::Problem; 
							  findactions::Function,
							  applyaction::Function,
							  goaltest::Function,
							  g::Function, 
							  h::Function, 
							  solution::Function,
							  kwargs...)
		@debug "\n" * "#"^29 * "\n# Astar tree search # \n" * "#"^29
		node = p.rootnode
		frontier = DataStructures.PriorityQueue{typeof(node), typeof(node.path_cost)}() 
		enqueue!(frontier, node, node.path_cost + h(p, node; action=nothing, kwargs...))
		@debug "Current queue: $(frontier)"
		while !isempty(frontier) 
			node = dequeue!(frontier)                                          
			if goaltest(p, node; kwargs...)
				@warn "Solution found!"
				return node, solution(node)
			end	
			@debug "Current node: $(node), current queue: $(frontier)"
			@debug "Possible action: $(findactions(p, node))"
			for action in findactions(p, node)
				@debug "possible actions: $(findactions(p, node))"
				newstate = applyaction(p, node, action; kwargs...)
				child = typeof(node)(newstate, 
									 node, 
									 action, 
									 g(p, node, newstate; kwargs...), 
									 node.depth + 1)
				@debug "new child: $(child)"
				enqueue!(frontier, child, child.path_cost + h(p, child; kwargs...))
			end
		end
		@warn "Failed to find a solution!"
		return nothing
	end

	"""
		Astargraphsearch(p::Problem; 
						findactions::Function, 
						applyaction::Function, 
						goaltest::Function, 
						g::Function, 
						h::Function, 
						solution::Function,
						kwargs...)

	Generic implementation of A* graph search. Determines sequence of actions.

	## keyword arguments:
		- findactions::Function: get available actions from a state 
		- applyaction::Function: get new node from applying an action on a state
		- goaltest::Function: evaluate if a node is in a goalstate
		- g(p, node, newstate)::Function: calculate the path cost from one node to the next state
		- h(p, node)::Function: calculate the heuristic for a current state towards a goal state
		- solution(node)::Function: trace the set of actions required to end up at the solution
		- kwargs: additions keyword argument pairs

	## Notes:
	to allow for extra output during the tree traversal, you need to enable a logger that can actually log the debug level.
	"""
	function Astargraphsearch(p::Problem; 
							  findactions::Function,
							  applyaction::Function,
							  goaltest::Function,
							  g::Function, 
							  h::Function, 
							  solution::Function,
							  kwargs...)
		@debug "\n" * "#"^29 * "\n# Astar graph search # \n" * "#"^29
		node = p.rootnode
		frontier = DataStructures.PriorityQueue{typeof(node), typeof(node.path_cost)}() 
		enqueue!(frontier, node, node.path_cost + h(p, node; action=nothing, kwargs...))
		@debug "Current queue: $(frontier)"
		explored = Set{typeof(node.state)}()
		while !isempty(frontier) 
			node = dequeue!(frontier)                                          
			if goaltest(p, node; kwargs...)
				@warn "Solution found!"
				return node, solution(node)
			end
			push!(explored, node.state)
			
			@debug "Current node: $(node), current queue: $(frontier)"
			@debug "Possible action: $(findactions(p, node))"
			for action in findactions(p, node)
				@debug "possible actions: $(findactions(p, node))"
				newstate = applyaction(p, node, action; kwargs...)
				child = typeof(node)(newstate, 
									 node, 
									 action, 
									 g(p, node, newstate; kwargs...), 
									 node.depth + 1)
				@debug "new child: $(child)"
				if child.state ∉ explored
					enqueue!(frontier, child, child.path_cost + h(p, child; kwargs...))
				end
			end
		end
		@warn "Failed to find a solution!"
		return nothing
	end
end;

# ╔═╡ 00e6a61a-e272-443e-b8f4-2a3948907713
begin
	# Supporting types and functions
	
	struct TwodimensionalNode <: Node
		state::Tuple{Int,Int}
		parent::Union{TwodimensionalNode, Nothing}
		action::Union{Nothing, Symbol}
		path_cost::Int
		depth::Int
	end
	
	struct TwodmimensionalGridproblem <: Problem
		rootnode::TwodimensionalNode
		goalstate::Tuple{Int,Int}
	end

	findactions_twodimensional(p::TwodmimensionalGridproblem, n::TwodimensionalNode) = [:left,:right,:up,:down]
	goaltest_twodimensional(p::TwodmimensionalGridproblem, n::TwodimensionalNode) = n.state == p.goalstate
	
	function applyaction_twodimensional(p::TwodmimensionalGridproblem, n::TwodimensionalNode, action::Symbol; kwargs...)
		if isequal(action, :up)
			return (n.state[1], n.state[2] + 1)
		elseif isequal(action, :down)
			return (n.state[1], n.state[2] - 1)
		elseif isequal(action, :left)
			return (n.state[1] - 1, n.state[2])
		elseif isequal(action, :right)
			return (n.state[1] + 1, n.state[2])
		end
	end

	function solution_twodimensional(n::TwodimensionalNode)
		actions = Vector{typeof(n.action)}()
		while !isnothing(n.parent)
			push!(actions, n.action)
			n = n.parent
		end
		return reverse!(actions)
	end

	# total path cost of going from a node to a new state
	g_2D(p::TwodmimensionalGridproblem, n::TwodimensionalNode, s::Tuple{Int,Int}; kwargs...) = n.path_cost + 1
	h_2D(p::TwodmimensionalGridproblem, n::TwodimensionalNode; kwargs...) = abs(p.goalstate[1] - n.state[1]) + abs(p.goalstate[2] - n.state[2])

	nothing
end

# ╔═╡ 8d6b647d-f72b-43e2-865a-fbcb27c409aa
begin
	Logging.disable_logging(LogLevel(5000))
	# Initialise the problem
	rootnode_2D = TwodimensionalNode((0,0), nothing, nothing,0,0)
	goal_2D = (1,3)
	prob_2D = TwodmimensionalGridproblem(rootnode_2D, goal_2D)
end

# ╔═╡ 84a58aa5-8830-4ff6-8e82-66ba837162fe
Astargraphsearch(prob_2D, 
				 findactions=findactions_twodimensional,
				 applyaction=applyaction_twodimensional,
				 goaltest=goaltest_twodimensional,
				 g=g_2D,
				 h=h_2D,
				 solution=solution_twodimensional)

# ╔═╡ ab5dea5e-8627-4601-bfce-12783cb6424a
Astartreesearch(prob_2D, 
				 findactions=findactions_twodimensional,
				 applyaction=applyaction_twodimensional,
				 goaltest=goaltest_twodimensional,
				 g=g_2D,
				 h=h_2D,
				 solution=solution_twodimensional)

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

The objective of this application is to implement different heuristics and compare the performance of the A-star algorithm.

Below you find a number of generic and specific functions that allow you to identify the possible actions for a given state and to determine the next state from a previous one. For didactic purposes, there also is a function that allows you to visualize the current state.
"""

# ╔═╡ 3b3eef7f-7747-4c34-b202-1f7fb99ddde5
begin
	struct NPuzzleNode{N} <: Node
		state::NTuple{N, Int64}
		parent::Union{Nothing,NPuzzleNode{N}}
		action::Union{Nothing, Symbol}
		path_cost::Int
		depth::Int
	end
	
	struct NPuzzle{N} <: Problem
		rootnode::NPuzzleNode
		goalstate::NTuple{N,Int64} 
		function NPuzzle(r::NPuzzleNode)
			round(sqrt(length(r.state))) == sqrt(length(r.state)) ? nothing : throw(DimensionMismatch("Npuzzle grid should be square!"))
			return new{length(r.state)}(r, Tuple(i for i in 0:length(r.state)-1))
		end
	end

	goaltest_npuzzle(p::NPuzzle, n::NPuzzleNode; kwargs...) = n.state == p.goalstate

	function findactions_npuzzle(p::NPuzzle{N}, n::NPuzzleNode{N}; kwargs...) where {N}
		d = Int(sqrt(N))
		possible_actions = [:left,:right,:up,:down]
		ind = findfirst(x->iszero(x), n.state) # find location of empty case
		# remove invalid options
		if ind <= d
			filter!(x -> x ≠ :up, possible_actions)
		end
		if ind > N - d
			filter!(x -> x ≠ :down, possible_actions)
		end
		if ind % d  == 1
			filter!(x -> x ≠ :left, possible_actions)
		end
		if ind % d  == 0
			filter!(x -> x ≠ :right, possible_actions)
		end

		return possible_actions
	end

	function applyaction_npuzzle(p::NPuzzle{N}, n::NPuzzleNode{N}, action::Symbol; ops::Dict, kwargs...) where {N}
		# make copy of state
		state = collect(n.state)
		# find location of empty case
		ind = findfirst(x->x==0, state)
		# index shifting
		state[ind], state[ind + ops[action]] = state[ind + ops[action]], state[ind]
		
		return Tuple(state)
	end

	# total path cost of going from a node to a new state
	g_npuzzle(p::NPuzzle{N}, n::NPuzzleNode{N}, s::NTuple{N,Int}; kwargs...) where {N}= n.path_cost + 1
	
	h_npuzzle(p::NPuzzle, n::NPuzzleNode; kwargs...) = 0 # no heuristic

	function solution_npuzzle(n::NPuzzleNode)
		actions = Vector{typeof(n.action)}()
		while !isnothing(n.parent)
			push!(actions, n.action)
			n = n.parent
		end
		return reverse!(actions)
	end

	"""
		prettyprint(n::NPuzzleNode{N})

	Supporting function to visualize the playing field
	"""
	function prettyprint(n::NPuzzleNode{N}) where {N}
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
		msg = "\n"
		for i = 1:d
			msg = msg *"| " *  prod(["$(morph(n.state[(i-1)*d+j])) | " for j in 1:d]) * "\n"
		end
		return msg
	end

	nothing
end

# ╔═╡ 522c9a02-12e2-480f-9b56-121f1e249411
let
	# some examples to illustrate
	Logging.disable_logging(LogLevel(-1000))
	@warn "DEMO LOCATIONS AND APPLYING ACTIONS"
	for state in [(0,1,2,3,4,5,6,7,8),
				  (1,0,2,3,4,5,6,7,8),
				  (1,2,0,3,4,5,6,7,8),
				  (4,1,2,0,3,5,6,7,8),
				  (4,1,2,3,0,5,6,7,8),
				  (4,1,2,3,5,0,6,7,8),
				  (4,1,2,3,5,6,0,7,8),
				  (4,1,2,3,5,6,7,0,8),
				  (4,1,2,3,5,6,7,8,0)]
		node = NPuzzleNode(state,nothing,nothing,0,0)
		prob = NPuzzle(node)
		actions = findactions_npuzzle(prob, node)

		# dictionary used for shifting locations (define once for a game)
		ops = Dict(:up=>-Int(sqrt(typeof(node).parameters[1])),
				   :down=>Int(sqrt(typeof(node).parameters[1])), 
				   :left=>-1,
				   :right=>1)

		action = rand(actions)
		newstate = applyaction_npuzzle(prob, node, action, ops=ops)
		newnode = NPuzzleNode(newstate,nothing,nothing,1,1)
		@info "Before:$(prettyprint(node))Valid actions: $(actions)\nchoosen action: $(action)\nAfter:$(prettyprint(newnode))"
	end

	
	@warn "DEMO OF A SMALL GAME"
	startnode = NPuzzleNode(Tuple([3, 1, 2, 4, 0, 5, 6, 7, 8]), nothing, nothing, 0,0)
	p = NPuzzle(startnode)
	startlayout = prettyprint(p.rootnode)
	@info "start:$(startlayout)"
	actions = [:left, :up]
	node = startnode
	ops = Dict(:up=>-Int(sqrt(typeof(node).parameters[1])),
				   :down=>Int(sqrt(typeof(node).parameters[1])), 
				   :left=>-1,
				   :right=>1)
	for action in actions
		newstate = applyaction_npuzzle(p, node, action, ops=ops)
		node = NPuzzleNode(newstate,node,action,node.path_cost+1,node.depth+1)
		@info "after action $(action): $(prettyprint(node))"
	end	
end

# ╔═╡ 54d03898-427b-41c4-9fbe-f9623cbd12e5
md"""
The actual problem we need to solve is defined below:
"""

# ╔═╡ a62d390c-cc13-4797-83a8-03e4f370ec49
begin
	puzzlerootnode = NPuzzleNode((7,2,4,5,0,6,8,3,1), nothing, nothing, 0, 0)
	my9puzzle = NPuzzle(puzzlerootnode)
	ops = Dict(:up=>-Int(sqrt(typeof(puzzlerootnode).parameters[1])),
			   :down=>Int(sqrt(typeof(puzzlerootnode).parameters[1])), 
			   :left=>-1,
			   :right=>1)
end

# ╔═╡ 41e2d7d9-96b1-4c87-afb3-b5e6bb48f715
begin
	# defining the heuristics

	# h1 - number of misplaced tiles
	h1_npuzzle(p::NPuzzle, n::NPuzzleNode; kwargs...) = sum(p.goalstate .!= n.state) 

	# h2 - sum of manhattan distances
	const Xcoord = [1 2 3 1 2 3 1 2 3]
	const Ycoord = [1 1 1 2 2 2 3 3 3]
	
	function h2_npuzzle(p::NPuzzle, n::NPuzzleNode; kwargs...)
		sum(abs(Xcoord[s+1] - Xcoord[g+1]) + abs(Ycoord[s+1] - Ycoord[g+1]) for (s,g) in Iterators.zip(n.state, (0,1,2,3,4,5,6,7,8)))
	end
	
	h1_npuzzle(my9puzzle, puzzlerootnode), h2_npuzzle(my9puzzle, puzzlerootnode)
end

# ╔═╡ a740dcb0-07c1-4c22-8579-f49b13e2e456
let
	Logging.disable_logging(LogLevel(-1000))
	(lastnode, moves) = Astargraphsearch(my9puzzle, 
				 findactions=findactions_npuzzle,
				 applyaction=applyaction_npuzzle,
				 goaltest=goaltest_npuzzle,
				 g=g_npuzzle,
				 h=h1_npuzzle,
				 solution=solution_npuzzle,
				 ops=ops)
	moves
end


# ╔═╡ 33505da9-88de-405a-970a-afe71d234c59
let
	Logging.disable_logging(LogLevel(-1000))
	(lastnode, moves) = Astargraphsearch(my9puzzle, 
				 findactions=findactions_npuzzle,
				 applyaction=applyaction_npuzzle,
				 goaltest=goaltest_npuzzle,
				 g=g_npuzzle,
				 h=h2_npuzzle,
				 solution=solution_npuzzle,
				 ops=ops)
	moves
end

# ╔═╡ b35e5664-7143-11eb-1e75-17b00177d1f9
md"""
# Additional problems
We have seen several search algorithms, each time illustrated by one or more examples. The difficulty of this kind of problems is not so much in the implementation of an algorithm, but in formalising a problem and choosing a correct heuristic. The following problems will illustrate this.
"""

# ╔═╡ a653d9a8-7143-11eb-3812-396c3ec732bd


# ╔═╡ 28f0121c-713c-11eb-2aff-e7a9c592fa53
md"""

## TSP
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

# ╔═╡ f972052d-ba4f-4d16-8d2c-c856466e29e8
begin
	# some visualisation functions
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
	function plotgraph(g::SimpleWeightedGraph, coords::Vector{NTuple{2,Float64}}; wf::Real=5,draw_MST::Bool=true, draw_weights::Bool=false)
		p = plot(legend=:bottomright)
		for i in 1:length(coords)
			annotate!([coords[i][1]], [coords[i][2]] .+0.05, "$(i)")
		end
		for e in edges(g)
			plot!([coords[e.src][1], coords[e.dst][1]],[coords[e.src][2], coords[e.dst][2]], 
				markershape=:circle, linewidth=e.weight*wf, color=:black, markeralpha = 0.5,
				linealpha=0.2, label="", markercolor=:black, markersize=5)
			if draw_weights
				annotate!([(coords[e.src][1] + coords[e.dst][1])/2], [(coords[e.src][2] + coords[e.dst][2])/2],"$(e.weight)", color=:blue);
			end
		end
		plot!([],[],markershape=:circle, color=:black,
				linealpha=0.2, label="Full network", markercolor=:black, markersize=5, markeralpha = 0.5)
		
		if draw_MST
			MST = prim_mst(g, weights(g))
			for e in MST
				plot!([coords[e.src][1], coords[e.dst][1]],[coords[e.src][2], coords[e.dst][2]], 
						markershape=:circle, linewidth=weights(g)[e.src,e.dst]*wf, color=:red,
						linealpha=0.8, label="", linestyle=:solid, markeralpha = 0)

			end
			plot!([],[],linealpha=0.8, label="MST", color=:red, linestyle=:solid)
		end	

		# get domain bounds
		xmin = mapreduce(x->x[1], min, coords)
		xmax = mapreduce(x->x[1], max, coords)
		ymin = mapreduce(x->x[2], min, coords)
		ymax = mapreduce(x->x[2], max, coords)
		xlims!(xmin*0.8, xmax*1.2)
		ylims!(ymin*0.8, ymax*1.2)
		plot!(legend=:best)
		return p
	end

	"""
		plotsol(path::Array{Int64,1}, coords::Vector{NTuple{N, Float64}})

	Function that plots the final TSP solution
	"""
	function plotsol(path::Array{Int64,1}, coords::Vector{NTuple{2, Float64}})
		#settings = Dict(:arrow => :arrow, :lw => 2, :color=>:blue, :label=>"")
		p = plot()
		for i = 1:length(path)-1
			plot!([coords[path[i]][1]; coords[path[i+1]][1]],
			  	  [coords[path[i]][2]; coords[path[i+1]][2]],
			  arrow=:arrow, color=:blue,label="")
		end
		# add final path
		plot!([coords[path[end]][1]; coords[path[1]][1]],
			  [coords[path[end]][2]; coords[path[1]][2]],
			  arrow=:arrow, color=:blue,label="")
		
		# get domain bounds
		xmin = mapreduce(x->x[1], min, coords)
		xmax = mapreduce(x->x[1], max, coords)
		ymin = mapreduce(x->x[2], min, coords)
		ymax = mapreduce(x->x[2], max, coords)
		xlims!(xmin*0.8, xmax*1.2)
		ylims!(ymin*0.8, ymax*1.2)
		
		return p
	end
end

# ╔═╡ fa7eb226-43b6-4560-9841-313843a529d6
md"""
## Pathfinding in a maze
Reconsider the maze problem from last time. Look into the added value of using the A$^*$ algorithm. What heuristics do you use? Compare between them and analyse generalizability.
"""

# ╔═╡ f5c09c78-f2b9-4a65-86a5-99e46f9c84e9


# ╔═╡ b64a41e5-6f06-442b-bd6d-4d04d2520208
md"""
## Pathfinding for (military) robotics
Terrestrial platform mobility is different from that of aerial platform,  because you need to account for the complex geographical elements such as mountainous and potamic terrain. It is also different from intelligent robot mobility on a given plane, as safety factor in simulated military countermining environment should be calculated in terrestrial platform.

Consider the following configuration:
* a square world 10x10km
* intial location: (0,2.5)
* target location (10,1)
* Each path between two parts of the terrain has a passableness rating ``t(A,B) \in [0,1]``, where 0 denotes unpassable terrain. For this application, consider ``t(x,y) = \cases{\frac{1}{\sqrt{0.1x} + 0.1y + 1} \;\text{ for } |x-4|\ge0.4, |y-4| \le 0.1) \\ 0\; (x=4, y\ne4)}``
* Each path between two parts of the terrain has a safety measurement ``s(A,B) \in [0,1]``, where 0 denotes very unsafe terrain (and thus to be avoided). For this application, consider ``s(x,y) = \cases{|x - 2.5||y-2.5| \text{ for }|x - 2.5||y-2.5| < 1) \\ 1 \text{ elsewhere} }``

* This allows us to define the cost of a path as follows: ``c(A,B) = \cases{\frac{|AB|}{t(A,B)^\alpha s(A,B)^\beta}  \text{ if both } t(A,B) \ne 0 \text{ and } s(A,B) \ne 0 \\ +\infty\text{ else}}``. In this expression, the constants ``\alpha`` and ``\beta`` can be used to modulate the importance of each contribution. If one of these values is zero, it is not taken into consideration.

#### Tasks:
* Think about how you can use A-star to find the optimal route to the target.
* Implement this and find solution
* Analyse the impact of the values of ``\alpha`` and ``\beta`` on the solution. Do the different solutions make sense?
"""



# ╔═╡ d9fdfde6-057f-4fe4-9821-c69eec943f37
begin
	source = (0,  2.5)
	target = (10, 1)
	
	x = range(0, 10, length=201)
	y = range(0, 10, length=201)
	Sx = findfirst(x->x==source[1], x)
	Sy = findfirst(x->x==source[2], y)
	Tx = findfirst(x->x==target[1], x)
	Ty = findfirst(x->x==target[2], y)
	
	function t(x,y)
		if abs(x - 4)>0.4 || abs(y - 4) < 0.1
			return 1 / (sqrt(0.1x) + 0.1y + 1)
		else
			return zero(typeof(x))
		end
	end
	
	s(x,y) = abs(x-2.5) * abs(y-2.5) < 1 ? abs(x-2.5) * abs(y-2.5) : one(typeof(x))
	
	A = reshape(map(v->s(v...), Iterators.product(x,y)), length(x),:)
	B = reshape(map(v->t(v...), Iterators.product(x,y)), length(x),:)
	nothing
end

# ╔═╡ 49d11f58-713c-11eb-20bb-e3a3dc889b6b
begin
	struct TSPState
		visited::Set{Int64}
		trip::Array{Int64}
	end
	
	struct TSPNode <: Node
		state::TSPState
		parent::Union{Nothing, TSPNode}
		action::Union{Nothing, Int64}
		path_cost::Float64
		depth::Int
	end
	
	struct TSPProblem <: Problem
		rootnode::TSPNode
		goalstate::Set{Int64}
		graph::SimpleWeightedGraph
		coords::Vector{Tuple{Float64,Float64}}
	end

	TSPProblem(n::Int) =  TSPProblem([(rand(), rand()) for _ in 1:n])
	
	function TSPProblem(x::Vector{Tuple{Float64,Float64}})
		A = zeros(length(x),length(x))
		for i = 1:size(A,1)
			for j = 1:size(A,1)
				A[i,j] = sqrt((x[i][1] - x[j][1])^2 + (x[i][2] - x[j][2])^2)
			end
		end
		return TSPProblem(A,coords=x)	
	end
	
	function TSPProblem(A::Matrix; 
						coords::Vector{Tuple{Float64,Float64}}=[(rand(),rand()) for _ in 1:size(A,1)])
		G = SimpleWeightedGraph(Float64.(issymmetric(A) ? A : A + A'))
		return TSPProblem(G, coords=coords)
	end

	function TSPProblem(G::Union{SimpleWeightedGraph,SimpleWeightedDiGraph};
				        coords::Vector{Tuple{Float64,Float64}}=[(rand(),rand()) for _ in 1:size(A,1)])
		rootstate = TSPState(Set(1),[1])
		rootnode = TSPNode(rootstate, nothing, nothing, 0, 0)
		goalstate = Set(vertices(G))
		return TSPProblem(rootnode, goalstate, G, coords)
	end

	goaltest_TSP(p::TSPProblem, n::TSPNode; kwargs...) = n.state.visited == p.goalstate

	findactions_TSP(p::TSPProblem, n::TSPNode; kwargs...) = setdiff(p.goalstate, n.state.visited)

	applyaction_TSP(p::TSPProblem, n::TSPNode, action::Int64; kwargs...) = TSPState(push!(copy(n.state.visited), action), 
			 push!(copy(n.state.trip), action))
	
	
	g_TSP(p::TSPProblem, n::TSPNode, s::TSPState; kwargs...) = n.path_cost + p.graph.weights[s.trip[end], n.state.trip[end]] # note that weights are indexed by [destination, source] in SimpleWeightedGraphs

	function h_TSP_straightline(p::TSPProblem, n::TSPNode; action=nothing, kwargs...)
		if isnothing(action)
			return 0.
		else
			return sqrt((p.coords[n.state.trip[end]][1] - p.coords[n.state.trip[end-1]][1]) ^2 +
		 	 			(p.coords[n.state.trip[end]][2] - p.coords[n.state.trip[end-1]][2]) ^2)
		end
	end
	
	function h_TSP_MST(p::TSPProblem, n::TSPNode; action=nothing, kwargs...) 
		remaining = collect(union(setdiff(p.goalstate, n.state.visited), n.state.trip[1]))
		if length(remaining) <= 1
			return zero(eltype(p.graph.weights))
		else
			limited_graph = SimpleWeightedGraph(p.graph.weights[remaining,remaining])
			mst = prim_mst(limited_graph)
			return sum(limited_graph.weights[e.dst, e.src] for e in mst)
		end
	end

	solution_TSP(n::TSPNode) = n.state.trip
end

# ╔═╡ 5104f015-3bb8-4a1a-a81c-2e06d7b43ca1
begin
	dm = Float64[0 10 15 20;
		 10 0 35 25;
		 15 35 0 30;
		 20 25 30 0]
	G = SimpleWeightedGraph(dm)
	TSProb = TSPProblem(dm)
end

# ╔═╡ 38e1e485-2f23-4bf4-9c65-efa7426324f9
let
	# using straight line distance
	Logging.disable_logging(LogLevel(-1000))
	(lastnode, moves) = Astargraphsearch(TSProb, 
				 findactions=findactions_TSP,
				 applyaction=applyaction_TSP,
				 goaltest=goaltest_TSP ,
				 g=g_TSP,
				 h=h_TSP_straightline,
				 solution=solution_TSP)
end

# ╔═╡ 9e924a04-f4f3-4228-a3ae-465ff191d884
begin
	# using MST
	Logging.disable_logging(LogLevel(-1000))
	(lastnode, moves) = Astargraphsearch(TSProb, 
				 findactions=findactions_TSP,
				 applyaction=applyaction_TSP,
				 goaltest=goaltest_TSP ,
				 g=g_TSP,
				 h=h_TSP_MST,
				 solution=solution_TSP)
end

# ╔═╡ adaef3a5-f845-488b-84b1-64215d4713be
begin
	p1 = plotgraph(TSProb.graph, TSProb.coords, wf=1)
	title!("TSP problem and MST")
	p2 = plotsol(moves, TSProb.coords)
	title!("TSP solution")
	plot(p1,p2)
end

# ╔═╡ 99bbb244-713c-11eb-34be-d78f80834d46
# small illustration
let
	Logging.disable_logging(LogLevel(-1000))
	myproblem = TSPProblem(10)
	
	(lastnode, moves) = Astargraphsearch(myproblem, 
				 findactions=findactions_TSP,
				 applyaction=applyaction_TSP,
				 goaltest=goaltest_TSP ,
				 g=g_TSP,
				 h=h_TSP_straightline,
				 solution=solution_TSP)
	
	p1 = plotgraph(myproblem.graph, myproblem.coords, wf=3)
	title!("TSP problem and MST")
	p2 = plotsol(moves, myproblem.coords)
	title!("TSP solution")
	plot(p1,p2)
end

# ╔═╡ 62a941d2-f5a2-4e59-a315-91886d920152
begin
	heatmap(A')
	xticks!(collect(1:20:length(x)), ["$(v)" for v in collect(x)[1:20:length(x)]])
	yticks!(collect(1:20:length(y)), ["$(v)" for v in collect(x)[1:20:length(y)]])
	title!("danger zone")
	scatter!([Sx],[Sy],label="Start",color=:Lightblue)
	scatter!([Tx],[Ty],label="Target",color=:green)
	xlims!(1, size(A,2))
	ylims!(1, size(A,1))
end

# ╔═╡ 09af0731-cb20-4331-9503-03f691cde775
begin
	heatmap(B')
	xticks!(collect(1:20:length(x)), ["$(v)" for v in collect(x)[1:20:length(x)]])
	yticks!(collect(1:20:length(y)), ["$(v)" for v in collect(x)[1:20:length(y)]])
	title!("passable zone")
	scatter!([Sx],[Sy],label="Start",color=:Lightblue)
	scatter!([Tx],[Ty],label="Target",color=:green)
	xlims!(1, size(B,2))
	ylims!(1, size(B,1))
end

# ╔═╡ 6246df51-2fc4-47be-bd0e-8c7ed0aabed1
begin
	"""
		Astargraphsearch(p::Problem; 
						findactions::Function, 
						applyaction::Function, 
						goaltest::Function, 
						g::Function, 
						h::Function, 
						solution::Function,
						kwargs...)

	Generic implementation of faster A* graph search. Determines sequence of actions.

	## keyword arguments:
		- findactions::Function: get available actions from a state 
		- applyaction::Function: get new node from applying an action on a state
		- goaltest::Function: evaluate if a node is in a goalstate
		- g(p, node, newstate)::Function: calculate the path cost from one node to the next state
		- h(p, node)::Function: calculate the heuristic for a current state towards a goal state
		- solution(node)::Function: trace the set of actions required to end up at the solution
		- kwargs: additions keyword argument pairs

	## Notes:
	to allow for extra output during the tree traversal, you need to enable a logger that can actually log the debug level.
	"""
	function fastAstargraphsearch(p::Problem; 
							  findactions::Function,
							  applyaction::Function,
							  goaltest::Function,
							  g::Function, 
							  h::Function, 
							  solution::Function,
							  kwargs...)
		@debug "\n" * "#"^29 * "\n# Faster Astar graph search # \n" * "#"^29
		node = p.rootnode
		frontier = DataStructures.PriorityQueue{typeof(node), typeof(node.path_cost)}() 
		enqueue!(frontier, node, node.path_cost + h(p, node; action=nothing, kwargs...))
		@debug "Current queue: $(frontier)"
		explored = Set{typeof(node.state)}()
		while !isempty(frontier) 
			node = dequeue!(frontier)                                          
			if goaltest(p, node; kwargs...)
				@warn "Solution found!"
				return node, explored, solution(node)
			end
			push!(explored, node.state)
			
			@debug "Current node: $(node), current queue: $(frontier)"
			@debug "Possible action: $(findactions(p, node))"
			for action in findactions(p, node)
				@debug "possible actions: $(findactions(p, node))"
				newstate = applyaction(p, node, action; kwargs...)
				child = typeof(node)(newstate,
									 action, 
									 g(p, node, newstate; kwargs...),
									 vcat(node.path, action),
									 node.depth + 1)
				@debug "new child: $(child)"
				if child.state ∉ explored
					enqueue!(frontier, child, child.path_cost + h(p, child; kwargs...))
				end
			end
		end
		@warn "Failed to find a solution!"
		return nothing
	end

	struct RobotNode <: Node
		state::NTuple{2,Float64}
		action::Union{Nothing, NTuple{2,Float64}}
		path_cost::Float64
		path::Vector{NTuple{2,Float64}}
		depth::Int
	end
	
	struct RobotProblem <: Problem
		rootnode::RobotNode
		goalstate::NTuple{2,Float64}
		domain::Vector{Float64} # L R B T
		Δx::Float64
		Δy::Float64
		actions#::Vector{NTuple{2,Float64}}
		tolpos::Float64
	end

	RobotProblem(start::NTuple{2,Float64}=(0.,0.), 
				 stop::NTuple{2,Float64}=(5.,5.),
				 bounds::Vector{Float64}=[min(start[1],stop[1]); 			 	
					 					  max(start[1],stop[1]); 				
					 					  min(start[1],stop[2])
				 						  max(start[2],stop[2])];
				 Δx::Float64=1.,Δy::Float64=1.) = RobotProblem(
								  RobotNode(start, nothing, 0., Vector{NTuple{2,Float64}}(), 0),
								  stop,
								 bounds,
								  Δx,
								  Δy,
								[(-Δx, Δy); (zero(typeof(Δy)), Δy); (Δx, Δy);
								 (-Δx, zero(typeof(Δy)));  (Δx,zero(typeof(Δy)));
								 (-Δx, -Δy);(zero(typeof(Δy)), -Δy); (Δx, -Δy)],
								sqrt(Δx^2+Δy^2))

	# goal test based on closeness
	goaltest_Robot(p::RobotProblem, n::RobotNode; kwargs...) = sqrt((n.state[1] - p.goalstate[1])^2 + (n.state[2] - p.goalstate[2])^2 )  <=  p.tolpos

	"""
	uses an additional filter function f to remove locations that are not passable
	"""
	function findactionsRobot(p::RobotProblem, n::RobotNode; 
							  f::Function=(node,step)->true, kwargs...) 
		inboundsteps = filter(x-> n.state[1] + x[1]>= p.domain[1] &&  
						  n.state[1] + x[1]<= p.domain[2] &&
						  n.state[2] + x[2]>= p.domain[3] &&  
						  n.state[2] + x[2]<= p.domain[4], p.actions)
		return filter!(x->f(n,x), inboundsteps)
	end

	# special filter function

	applyaction_Robot(p::RobotProblem, n::RobotNode, action::NTuple{2,Float64}; kwargs...) = (n.state[1] + action[1], n.state[2] + action[2])

	# path cost 
	g_Robot(p::RobotProblem, n::RobotNode, s::NTuple{2,Float64}; kwargs...) = n.path_cost + sqrt( (n.state[1] - s[1])^2 + (n.state[2] - s[2])^2)

	h_Robot(p::RobotProblem, n::RobotNode; kwargs...) = 	sqrt( (n.state[1] - p.goalstate[1])^2 + (n.state[2] - p.goalstate[2])^2)

	solution_Robot(n::RobotNode) = n

	# helper function for plotting
	function plotrobotproblem(p::RobotProblem; 
							  explored::Union{Nothing,Set{NTuple{2, Float64}}}=nothing,
							  solution::Union{Nothing,Vector{NTuple{2, Float64}}}=nothing)
		domain = Iterators.product(p.domain[1]:p.:Δx:p.domain[2], 
						           p.domain[3]:p.:Δy:p.domain[4])
		solpath = Vector{NTuple{2, Float64}}(undef, length(solution) + 1)
		solpath[1] = p.rootnode.state
		for i in eachindex(solution)
			solpath[i+1] = (solpath[i][1] + solution[i][1], 
							solpath[i][2] + solution[i][2])
		end
		E = !isnothing(explored) ? map(x->x∈explored, domain) : map(x->false, domain)
		S = !isnothing(solution) ? map(x->x∈solpath, domain) : map(x->false, domain)

		return E,S, solpath
	end
	
end

# ╔═╡ 655dcb12-c5ee-457e-8329-1d0ad0323fec
let
	prob = RobotProblem( (0., 0.), (10., 1.),Δx=1., Δy=1.)
	# only small passage in middle
	function blocker(n::RobotNode, a::NTuple{2,Float64})
		if n.state[1] + a[1] >= 4 && n.state[1] + a[1] <= 6 && 
		   n.state[2] + a[2] >= 4 && n.state[2] + a[2] <= 6 
			return true
		else
			return false
		end
	end

	fn, vis, _ = fastAstargraphsearch(prob, findactions=findactionsRobot, applyaction=applyaction_Robot, goaltest=goaltest_Robot, g=g_Robot, h=h_Robot, solution=solution_Robot,f=blocker)

	A,B,C = plotrobotproblem(prob, explored=vis, solution=fn.path)
	plot(heatmap(A), heatmap(B))


end

# ╔═╡ dc66bf17-f869-42e9-b032-aedff51bba24
begin
	qq = RobotProblem()
	goaltest_Robot(qq, qq.rootnode)
	acts = findactionsRobot(qq,qq.rootnode)
	#applyaction_Robot(qq, qq.rootnode, acts[1])
end

# ╔═╡ d50dfa8f-7d64-4f87-97bb-7e6dcd3fc0a3
f, disc, ff = fastAstargraphsearch(qq, findactions=findactionsRobot, applyaction=applyaction_Robot, goaltest=goaltest_Robot, g=g_Robot, h=h_Robot, solution=solution_Robot)

# ╔═╡ 161bce8f-0f8e-4651-9d96-d997070c9eb0
plotrobotproblem(qq, explored=disc, solution=ff.path)

# ╔═╡ 16b44b5c-e4f8-423e-8138-659417a4f58d


# ╔═╡ Cell order:
# ╟─794171c4-ba1a-4d75-a3fa-b250d36e8041
# ╠═ea9468a2-710f-11eb-2284-6d8af448d9eb
# ╠═4d7293cb-1646-46c4-a324-e33f32915d85
# ╟─5690f528-fdbc-40c6-9a38-9fa20a36fb75
# ╠═49f2c47b-ae96-42bd-ac84-ebd63d410799
# ╠═4063607a-ea41-4ccc-90c0-5c729e620572
# ╠═00e6a61a-e272-443e-b8f4-2a3948907713
# ╠═8d6b647d-f72b-43e2-865a-fbcb27c409aa
# ╠═84a58aa5-8830-4ff6-8e82-66ba837162fe
# ╠═ab5dea5e-8627-4601-bfce-12783cb6424a
# ╟─7782b3b2-710f-11eb-1a92-e12b8b1f3299
# ╠═3b3eef7f-7747-4c34-b202-1f7fb99ddde5
# ╠═522c9a02-12e2-480f-9b56-121f1e249411
# ╟─54d03898-427b-41c4-9fbe-f9623cbd12e5
# ╠═a62d390c-cc13-4797-83a8-03e4f370ec49
# ╠═41e2d7d9-96b1-4c87-afb3-b5e6bb48f715
# ╠═a740dcb0-07c1-4c22-8579-f49b13e2e456
# ╠═33505da9-88de-405a-970a-afe71d234c59
# ╟─b35e5664-7143-11eb-1e75-17b00177d1f9
# ╠═a653d9a8-7143-11eb-3812-396c3ec732bd
# ╟─28f0121c-713c-11eb-2aff-e7a9c592fa53
# ╠═49d11f58-713c-11eb-20bb-e3a3dc889b6b
# ╠═5104f015-3bb8-4a1a-a81c-2e06d7b43ca1
# ╠═38e1e485-2f23-4bf4-9c65-efa7426324f9
# ╠═9e924a04-f4f3-4228-a3ae-465ff191d884
# ╟─f972052d-ba4f-4d16-8d2c-c856466e29e8
# ╠═adaef3a5-f845-488b-84b1-64215d4713be
# ╠═99bbb244-713c-11eb-34be-d78f80834d46
# ╟─fa7eb226-43b6-4560-9841-313843a529d6
# ╠═f5c09c78-f2b9-4a65-86a5-99e46f9c84e9
# ╟─b64a41e5-6f06-442b-bd6d-4d04d2520208
# ╠═d9fdfde6-057f-4fe4-9821-c69eec943f37
# ╟─62a941d2-f5a2-4e59-a315-91886d920152
# ╟─09af0731-cb20-4331-9503-03f691cde775
# ╠═6246df51-2fc4-47be-bd0e-8c7ed0aabed1
# ╠═655dcb12-c5ee-457e-8329-1d0ad0323fec
# ╠═dc66bf17-f869-42e9-b032-aedff51bba24
# ╠═d50dfa8f-7d64-4f87-97bb-7e6dcd3fc0a3
# ╠═161bce8f-0f8e-4651-9d96-d997070c9eb0
# ╠═16b44b5c-e4f8-423e-8138-659417a4f58d
