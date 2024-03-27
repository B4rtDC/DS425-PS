### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ f39ca040-710d-11eb-102e-c5521f152a14
begin 
	using Pkg
	Pkg.activate(joinpath(@__DIR__,".."))
	using PlutoUI
	using DataStructures
	using Logging
	using Graphs
	using Random
	Logging.global_logger(Logging.ConsoleLogger(stdout, LogLevel(-5000)  ) )
	Logging.disable_logging(LogLevel(-1000))
	
	const MAXLOG=100
	TableOfContents(title="Search")
end

# ╔═╡ f5517d51-2a4f-4159-9820-1d18a097c9e4
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

# ╔═╡ cf5232c8-710c-11eb-00bb-7d9bcefd8660
md"""
# Uninformed search
A problem consists of five parts: the **initial state**, a set of **actions**, a **transition model** describing the results of those actions, a **goal test function**, and a **path cost function**. The environment of the problem is represented by a state space. A path through the state space from the initial state to a goal state is a solution.


### Quick questions
* What is the difference between the state space and the search tree?
* How to determine what node from the frontier to expand next?
* What is a redundant path and how to deal with them?

### True or false? Argument or provide a counterexample
* breath-first search is cost-optimal
* depth-first search is cost-optimal
* uniform-cost search is cost-optimal


"""

# ╔═╡ f636e0b2-bcfe-457c-bf4e-a79eecae8f33
md"""
## Small problems
Determine the problem formulation for the following:

1. You start with the sequence EACEAABAAB, or in general any sequence made from A, B, C, and E. You can transform this sequence using the following equalities: AC = E, AB = BC, BB = E, and Ex = x for any x. For example, ABBC can be transformed into AEC, and then AC, and then E. Your goal is to produce the sequence E.

2. There are six glass boxes in a row, each with a lock. Each of the first five boxes holds a key unlocking the next box in line; the last box holds a banana. You have the key to the first box, and you want the banana.

## Computer work
For the first small problem, implement the breadth-first search algorithm and test it for different start sequences.

*Tip:* given the fact that breadth-first is based on a FIFO-queue, you might want to use [`DataStructures.jl`](https://juliacollections.github.io/DataStructures.jl/latest/), in particular, have a look a `Stack` and `Queue`.
"""

# ╔═╡ 5e9aaef6-75e0-11eb-209d-fb7e9c6f00a5
# something to get you started:
begin
	const comb = Dict(  "AC"=>"E",
                    "AB"=>"BC",
                    "BB"=>"E",
                    "EA"=>"A",
                    "EB"=>"B",
                    "EC"=>"C",
                    "EE"=>"E")
	
    abstract type Node end;
    abstract type Problem end;
	
	function goaltest_sequence() end;
	function findactions_sequence() end;
	function solution_sequence() end;

	function treesearch(p::Problem; method::Symbol=:breadthfirst,
									goaltest::Function, 
								    findactions::Function, 
									applyactions::Function,
									solution::Function) 
		
		if isequal(method, :breadthfirst)
			# create FIFO Queue
			frontier = DataStructures.Queue{typeof(p.rootnode)}()
		else
			throw(ArgumentError("""method \"$(method)\" not implemented yet"""))
		end
		
		rootnode = p.rootnode;
		# place root node in the frontier
		enqueue!(frontier, rootnode)                  
		while !isempty(frontier)
			node = dequeue!(frontier)
			for action in findactions(node)
				child = applyactions(node, action)
				if child ∉ frontier
					if goaltest(child)
						@warn "Solution found! :-)"
						return solution(child)
					end
					enqueue!(frontier, child)
				end
			end
		end
		@warn "failed to find a solution! :-("
	end
	
	nothing
end

# ╔═╡ ba72d9d4-9914-4410-8670-fb9e531f6403
begin

	struct SequenceAction 
		idx::Int
		sequence::String
	end
	
	struct SequenceNode <: Node
		state::String
		parent::Union{Nothing, SequenceNode} # other ideas: store sequence of actions
		action::Union{Nothing, SequenceAction}
		path_cost::Int
		level::Int
	end
	
	struct SequenceProblem <: Problem
		rootnode::Node
	end

	function findactions_sequence(n::SequenceNode)
		actions = Vector{SequenceAction}()
		for i = 1:length(n.state)-1
			if n.state[i:i+1] ∈ keys(comb)
				push!(actions, SequenceAction(i, n.state[i:i+1]))
			end
		end
		return actions
	end

	goaltest_sequence(n::SequenceNode) = n.state == "E"

	function applyactions(n::SequenceNode, action::SequenceAction)
		newstate =  n.state[1:action.idx-1] * comb[n.state[action.idx:action.idx+1]] * n.state[action.idx+2:end]
		return SequenceNode(newstate, n, action, n.path_cost+1, n.level+1)
	end

	function solution_sequence(n::SequenceNode)
		sol = Vector{SequenceAction}()
		while !isnothing(n.parent)
			push!(sol, n.action)
			n = n.parent
		end
		return reverse!(sol)
	end
	
end

# ╔═╡ bab17fcb-deae-4cd1-a2cb-abb03e6c18cc
begin
	rootnode = SequenceNode("EACEAABAAB", nothing, nothing, 0, 0)
	myproblem = SequenceProblem(rootnode)
	#actions = findactions_sequence(rootnode)
	#goaltest_sequence(rootnode)
	#applyactions(rootnode, actions[3])
end

# ╔═╡ 89eff486-7d69-4f6c-96c2-73aec37ae26d
treesearch(myproblem,goaltest=goaltest_sequence, 
					 findactions=findactions_sequence, 
					 applyactions=applyactions,
					 solution=solution_sequence)

# ╔═╡ 5116ffec-21eb-4f1a-a2fc-001a7da3c052
md"""
# Continuing from here
* What about graph search?
* What about other methods? E.g. uniform cost search, depth first search$\dots$

## Maze problem
Solve a maze by using different search methods. As an additional challenge, you can make an illustration of the nodes that get explorered over time (similar to the examples shown during the lectures).

Below you have a maze generator (from [Rosetta Code](https://rosettacode.org/wiki/Maze_generation#Julia)). You can use it to generate a maze of a given size. 
The maze is represented as a matrix of `0` and `1`, where `0` represents a wall and `1` a free space. 
The starting point is always at the top left corner and the goal at the bottom right corner. In terms of node ids,
you start in node `n_cols + 2` and want to get to node `n_cols * (n_rows - 1)  - 2`.
"""

# ╔═╡ 722235d3-3af6-42b9-9f48-1aad5343f310
begin

	# helper functions
	check(bound::Vector) = cell -> all([1, 1] .≤ cell .≤ bound)	
	neighbors(cell::Vector, bound::Vector, step::Int=2) =
    filter(check(bound), map(dir -> cell + step * dir, [[0, 1], [-1, 0], [0, -1], [1, 0]]))

	"""
		walk(maze::Matrix, nxtcell::Vector, visited::Vector=[])

	walker to run the maze
	"""
	function walk(maze::Matrix, nxtcell::Vector, visited::Vector=[])
	    push!(visited, nxtcell)
	    for neigh in shuffle(neighbors(nxtcell, collect(size(maze))))
	        if neigh ∉ visited
	            maze[round.(Int, (nxtcell + neigh) / 2)...] = 0
	            walk(maze, neigh, visited)
	        end
	    end
	    maze
	end

	"""
		maze(w::Int, h::Int)
	
	Generate a maze of width w and height h.
	"""
	function maze(w::Int, h::Int)
	    maze = collect(i % 2 | j % 2 for i in 1:2w+1, j in 1:2h+1)
	    firstcell = 2 * [rand(1:w), rand(1:h)]
	    return walk(maze, firstcell)
	end

	pprint(matrix) = for i = 1:size(matrix, 1) println(join(matrix[i, :])) end

	"""
		printmaze(maze)
	
	Using special characters, print the maze.
	"""
	function printmaze(maze)
	    walls = split("╹ ╸ ┛ ╺ ┗ ━ ┻ ╻ ┃ ┓ ┫ ┏ ┣ ┳ ╋")
	    h, w = size(maze)
	    f = cell -> 2 ^ ((3cell[1] + cell[2] + 3) / 2)
	    wall(i, j) = if maze[i,j] == 0 " " else
	        walls[Int(sum(f, filter(x -> maze[x...] != 0, neighbors([i, j], [h, w], 1)) .- [[i, j]]))]
	    end
	    mazewalls = collect(wall(i, j) for i in 1:2:h, j in 1:w)
	    pprint(mazewalls)
	end
	
	"""
	    getgraph(M::Matrix)
	
	Extract a graph from a matrix where a value of 1 is a wall and a value of 0 is a path
	"""
	function getgraph(M::Matrix)
	    nrows, ncols = size(M)
	    G = Graph(nrows*ncols)
	    for i = 1:nrows
	        for j = 1:ncols
	            if M[i,j] == 0
	                if i < nrows && M[i+1,j] == 0
	                    add_edge!(G,(i-1)*ncols+j,(i)*ncols+j)
	                end
	                if j < ncols && M[i,j+1] == 0
	                    add_edge!(G,(i-1)*ncols+j,(i-1)*ncols+j+1)
	                end
	            end
	        end
	    end
	    return G
	end

	nothing
end

# ╔═╡ f20f6498-41c1-47a9-8afd-71059c9656c8
begin 
	# make a maze
	M =  maze(5,2)
	# show it
	printmaze(M)
	# get the graph
	G = getgraph(M)
	# Find your way out!
end

# ╔═╡ 2e3ec15e-c6e8-4dc2-9ff2-626f48609a72
M

# ╔═╡ bf336276-cf7d-4aa7-9c47-c605886c517d
begin
startnode = size(M,2) + 2
targetnode = size(M,2) * (size(M,1) - 1) - 1
	(startnode, targetnode)
end

# ╔═╡ 3490f7c1-dcd6-44c4-ae4b-10a97eb2753e
Graphs.neighbors(G, 19)

# ╔═╡ 43339c44-2218-4a89-b187-437f82a7cf2e
begin
	struct MazeProblem
		initial_state::Int
		terminal_state::Int
		M::Matrix
		G
	end

	function MazeProblem(nrows, ncols)
			M =  maze(nrows, ncols)
			G = getgraph(M)
			startnode = size(M,2) + 2
			targetnode = size(M,2) * (size(M,1) - 1) - 1
			return MazeProblem(startnode, targetnode, M, G)
	end
	MazeProblem(3,2, [2 2; 1 3], SimpleGraph()), MazeProblem(3,2)

	goaltest(s::Int, p::MazeProblem) = s == p.terminal_state
	find_actions(s::Int, p::MazeProblem) = Graphs.neighbors(p.G, s)
	apply_action(s::Int, a::Int, p::MazeProblem) = a
end

# ╔═╡ 7a2653c8-6aa8-40c5-85a0-faf2c9a35a61
# Maze with only two positions
P = MazeProblem(1,2)

# ╔═╡ 24f40a62-b599-4cac-929a-c9deb622e23f
printmaze(P.M)

# ╔═╡ b392ef3b-61ae-4668-90cf-8c2c1b331bae
# possible actions for 'nodes' in the maze
[find_actions(i, P) for i in vertices(P.G)]

# ╔═╡ 48731acc-8789-4169-8ffa-a12ea0146b2b
md"""
## Water jug puzzle

Puzzles of this type ask how many steps of pouring water from one jug to another (until either one jug becomes empty or the other becomes full) are needed to reach a goal state, specified in terms of the volume of liquid that must be present in some jug or jugs.

In the example below, such a setting is given. Here we want to identify what steps to take to end up in a situation where the two largest jugs each containt four units.

#### Questions
- Try to come up with a solution for this problem using one of the uninformed search methods.
- Argument why some methods might be better suited than others
- How easy or hard is it to expand this problem to settings where:
    - the volumes have different values
    - the number of jugs is modified
    - the goal state is different



$(PlutoUI.LocalResource("./img/Water_pouring_puzzle.png", :width => 500, :align => "middle"))
"""

# ╔═╡ 46144033-1bf9-4504-ba4b-55c932aa3e20


# ╔═╡ feb35ef0-56de-4333-945e-37db4c2d4a39
md"""
## Scheduling
How would you approach the problem of scheduling from a search perspective? E.g. your exam roster.

#### Questions
- How would you represent states?
- How are transitions determined? What are possible challenges here?
- What about goal state(s)? Any other things you should keep in mind?
"""

# ╔═╡ 6f6cbb0e-0117-49b0-9fd3-55954e196068


# ╔═╡ Cell order:
# ╟─f5517d51-2a4f-4159-9820-1d18a097c9e4
# ╟─f39ca040-710d-11eb-102e-c5521f152a14
# ╟─cf5232c8-710c-11eb-00bb-7d9bcefd8660
# ╟─f636e0b2-bcfe-457c-bf4e-a79eecae8f33
# ╠═5e9aaef6-75e0-11eb-209d-fb7e9c6f00a5
# ╠═ba72d9d4-9914-4410-8670-fb9e531f6403
# ╠═bab17fcb-deae-4cd1-a2cb-abb03e6c18cc
# ╠═89eff486-7d69-4f6c-96c2-73aec37ae26d
# ╟─5116ffec-21eb-4f1a-a2fc-001a7da3c052
# ╟─722235d3-3af6-42b9-9f48-1aad5343f310
# ╠═f20f6498-41c1-47a9-8afd-71059c9656c8
# ╠═2e3ec15e-c6e8-4dc2-9ff2-626f48609a72
# ╠═bf336276-cf7d-4aa7-9c47-c605886c517d
# ╠═3490f7c1-dcd6-44c4-ae4b-10a97eb2753e
# ╠═43339c44-2218-4a89-b187-437f82a7cf2e
# ╠═7a2653c8-6aa8-40c5-85a0-faf2c9a35a61
# ╠═24f40a62-b599-4cac-929a-c9deb622e23f
# ╠═b392ef3b-61ae-4668-90cf-8c2c1b331bae
# ╟─48731acc-8789-4169-8ffa-a12ea0146b2b
# ╠═46144033-1bf9-4504-ba4b-55c932aa3e20
# ╟─feb35ef0-56de-4333-945e-37db4c2d4a39
# ╠═6f6cbb0e-0117-49b0-9fd3-55954e196068