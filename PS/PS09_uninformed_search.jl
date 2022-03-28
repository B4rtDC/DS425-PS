### A Pluto.jl notebook ###
# v0.18.4

using Markdown
using InteractiveUtils

# ╔═╡ f39ca040-710d-11eb-102e-c5521f152a14
begin 
	using Pkg
	Pkg.activate(joinpath(@__DIR__,".."))
	
	using DataStructures
	using Logging
	Logging.global_logger(Logging.ConsoleLogger(stdout, LogLevel(-5000)  ) )
	Logging.disable_logging(LogLevel(-1000))
	
	const MAXLOG=100
	nothing
end

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

# ╔═╡ Cell order:
# ╟─f39ca040-710d-11eb-102e-c5521f152a14
# ╟─cf5232c8-710c-11eb-00bb-7d9bcefd8660
# ╟─f636e0b2-bcfe-457c-bf4e-a79eecae8f33
# ╠═5e9aaef6-75e0-11eb-209d-fb7e9c6f00a5
