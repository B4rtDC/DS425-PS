### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ f39ca040-710d-11eb-102e-c5521f152a14
begin 
	using DataStructures
	using Logging
	const MAXLOG=100
end

# ╔═╡ cf5232c8-710c-11eb-00bb-7d9bcefd8660
md"""
# Uninformed search
A problem consists of five parts: the **initial state**, a set of **actions**, a **transition model** describing the results of those actions, a **goal test function**, and a **path cost function**. The environment of the problem is represented by a state space. A path through the state space from the initial state to a goal state is a solution

Determine the problem formulation for the following:

1. You start with the sequence EACEAABAAB, or in general any sequence made from A, B, C, and E. You can transform this sequence using the following equalities: AC = E, AB = BC, BB = E, and Ex = x for any x. For example, ABBC can be transformed into AEC, and then AC, and then E. Your goal is to produce the sequence E.
    * initial state: start sequence e.g. "ABABAECCEC"
    * actions: transform a pair
    * transition model: AC = E, AB = BC, BB = E & Ex = x for any x
    * goal test function: sequence == "E"
    * path cost function: number of iterations
2. There are six glass boxes in a row, each with a lock. Each of the first five boxes holds a key unlocking the next box in line; the last box holds a banana. You have the key to the first box, and you want the banana.
    * initial state: closed boxes
    * actions: open box, obtain key
    * transition model: next box can be opened with newly obtained key
    * goal test function: obtain banana
    * path cost function: number of actions
    
For the first problem, implement the breadth-first search algorithm and test it for different start sequences.

*Tip:* given the fact that breadth-first is based on a FIFO-queue, you might want to take a look at `DataStructures.jl`
"""

# ╔═╡ 5e9aaef6-75e0-11eb-209d-fb7e9c6f00a5
# to get you started:
let
	comb = Dict(  "AC"=>"E",
                    "AB"=>"BC",
                    "BB"=>"E",
                    "EA"=>"A",
                    "EB"=>"B",
                    "EC"=>"C",
                    "EE"=>"E")
	
    struct Node end;
    struct Problem end;
	
	function goaltest_sequence() end;
	function findactions_sequence() end;
	function solution_sequence() end;
	function treesearch(p::Problem; goaltest::Function, findactions::Function, solution::Function) 
		rootnode = p.rootnode();
		fringe = DataStructures.Queue{p.Node}()       # create FIFO Queue
		enqueue!(fringe, rootnode)                  # place root node in the Queue
		while !isempty(fringe)
			node = dequeue!(fringe)
			for action in findactions(node)
				child = Node()
				if child ∉ fringe
					if goaltest(p, child)
						return solution()
					end
					enqueue!(fringe, child)
				end
			end
		end
		@warn "failed to find a solution"
	end;
	

end

# ╔═╡ Cell order:
# ╠═f39ca040-710d-11eb-102e-c5521f152a14
# ╟─cf5232c8-710c-11eb-00bb-7d9bcefd8660
# ╠═5e9aaef6-75e0-11eb-209d-fb7e9c6f00a5
