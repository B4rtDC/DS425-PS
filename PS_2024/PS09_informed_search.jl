### A Pluto.jl notebook ###
# v0.19.40

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
* Depth-first search always expands at least as many nodes as ``A^{∗}`` search with an admissible heuristic.
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
4. Is h = |u - x| + |v - y| an admissible heuristic for a state at (u, v)?
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

Below you can find a small example of the MST calculation for a tiny network that can be used.
"""

# ╔═╡ a799f7dc-b026-4cd8-8704-10aaa6c52183
begin
	# Number of points
	n = 10
	# Generate n random points in the unit square
	P = [(rand(), rand()) for _ in 1:n]
	# Compute the distance matrix between the points
	D = zeros(length(P),length(P))
	for i = 1:size(D,1)
		for j = 1:size(D,1)
			D[i,j] = sqrt((P[i][1] - P[j][1])^2 + (P[i][2] - P[j][2])^2)
		end
	end
	# Obtain a fully connected, weighted graph from the distance matrix
	TSP_G = SimpleWeightedGraph(Float64.(issymmetric(D) ? D : (D + D') ./ 2))
	# Obtain the MST
	TSP_MST = prim_mst(TSP_G, weights(TSP_G))
	## Make a plot of the graphs and its MST
	p = plot(xlims=(-0.1,1.1), ylims=(-0.1,1.1))
	# add node ID
	for i in eachindex(P)
		annotate!(p, [P[i][1]], [P[i][2]] .+0.05, "$(i)")
	end
	# add graph edges
	for e in edges(TSP_G)
		plot!(p, [P[e.src][1], P[e.dst][1]],[P[e.src][2], P[e.dst][2]], 
				markershape=:circle, linewidth=e.weight*10, color=:black, markeralpha = 0.5,
				linealpha=0.2, label="", markercolor=:black, markersize=5)
	end
	plot!(p, [],[],markershape=:circle, color=:black,
				linealpha=0.2, label="Full network", markercolor=:black, markersize=5, markeralpha = 0.5)
	# add MST edges
	for e in TSP_MST
		plot!([P[e.src][1], P[e.dst][1]],[P[e.src][2], P[e.dst][2]], 
						markershape=:circle, linewidth=weights(TSP_G)[e.src,e.dst]*10, color=:red,
						linealpha=0.8, label="", linestyle=:solid, markeralpha = 0)
	end
	plot!(p, [],[],linealpha=0.8, label="MST", color=:red, linestyle=:solid)
	p
end

# ╔═╡ faff5447-62af-4641-a85d-6dd0c64a42cf


# ╔═╡ fa7eb226-43b6-4560-9841-313843a529d6
md"""
## Pathfinding in a maze
Reconsider the maze problem from last time (uninformed search). Look into the added value of using the A$^*$ algorithm. What heuristics do you use? Compare between them and analyse generalizability.
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
# ╠═a799f7dc-b026-4cd8-8704-10aaa6c52183
# ╠═faff5447-62af-4641-a85d-6dd0c64a42cf
# ╟─fa7eb226-43b6-4560-9841-313843a529d6
# ╠═f5c09c78-f2b9-4a65-86a5-99e46f9c84e9
# ╟─b64a41e5-6f06-442b-bd6d-4d04d2520208
# ╠═d9fdfde6-057f-4fe4-9821-c69eec943f37
# ╟─62a941d2-f5a2-4e59-a315-91886d920152
# ╟─09af0731-cb20-4331-9503-03f691cde775
# ╠═6246df51-2fc4-47be-bd0e-8c7ed0aabed1
