### A Pluto.jl notebook ###
# v0.19.40

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

# ╔═╡ 902eeb2e-71d2-11eb-3322-53451cb23bf2
# dependencies
begin
	using Pkg
	Pkg.activate(joinpath(@__DIR__,".."))
	
	using PlutoUI
	using DataStructures
	using Logging
	using BenchmarkTools
	using Plots#, StatsPlots
	#using PlutoUI
	TableOfContents()
end

# ╔═╡ 9236c53d-accf-44e4-9a95-f71cfda90c03
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

# ╔═╡ d0fa06da-71d1-11eb-2d00-97ea31bbb5ed
md"""
# Constraint Satisfaction Problems (CSP)
CSP's represent a state with a set of (variable, value) pairs and represent the conditions for a solution by a set of constraints on the variables.

## Quick questions
* What is required to define a CSP?
* What is “consistency” in the context of CSPs?
* Define inference within the context of CSPs and give an example
* Explain the idea behind backtracking search and its different steps in your own words

## True or false? Argument or provide a counterexample
* The domain of a discrete CSP variable can be infinite 
* The domain of a CSP cannot be continuous.
* The heuristics used in backtracking for variable and value selection are domain-specific.

## Small problems
1. How many solutions are there for the map-coloring problem in (Figure 5.1) using RGB? How many solutions if four colors are allowed? Two colors? What about $n$ colors?

   $(PlutoUI.LocalResource("./img/map.png", :width => 500, :align => "center"))

2. Consider the Hamiltonian tour: given a network of cities connected by roads, choose an order to visit all cities in a country without repeating any. Provide a CSP formulation for this problem.

3. Consider the problem of allocating rectangular rooms within a given predefined contour shape, while satisfying given adjacency and size constraints. This work is significant with respect to complex building structures, such as hospitals, schools, offices, etc. You need to find non-overlapping places in a large rectangle for a number of smaller rectangles.

   $(PlutoUI.LocalResource("./img/floorplan.gif", "width" => 500))

4. Consider the constraint graph with 8 nodes {A1, A2, A3, A4, H, T, F1, F2}. Ai is connected to Ai+1 for all i, each Ai is connected to H, H is connected to T, and T is connected to each Fi. An overview is show in the illustration below. Find a 3-coloring of this graph by hand using backtracking with
    - the variable order {A1, H, A4, F1, A2, F2, A3, T} 
    - the value order {R, G, B}
   $(PlutoUI.LocalResource("./img/smallnet.png", :width => 150, :align => "center"))
   Compare your result with the result obtained when using forward checking as a filtering technique. What are your conclusions? What else could be done to improve performance?

"""

# ╔═╡ 1a8ce0dd-bece-437e-b44f-a31f8c155ce1
md"""
Solutions:
$(@bind sols PlutoUI.MultiCheckBox(["backtracking"; "backtracking with forward checking"]))
"""

# ╔═╡ 4b510a0c-55bb-4e28-8dcd-993ee8ef433f
md"""
**backtracking**:

$("backtracking" ∈ sols ? PlutoUI.LocalResource("./img/CSP_support.001.png", :width => 800, :align => "center") : "solution not shown")


**backtracking with forward selection**:

$("backtracking with forward checking" ∈ sols ? PlutoUI.LocalResource("./img/CSP_support.002.png", :width => 800, :align => "center") : "solution not shown")

"""

# ╔═╡ f95662cb-7034-4efd-916b-57c79cd8929c
md"""
## Implementation
Below you can find a generic implementation for backtracking search. Analyse the different components and make sure you understand the bigger picture.
"""

# ╔═╡ 6727b2df-d549-459b-9332-c7b607f09df2
md"""
### General CSP definition
"""

# ╔═╡ 82d6b21c-5a95-4cdb-bd0b-e061dec50169
begin
	abstract type AbstractCSP end;

	"""
		CSP
	
	CSP is a Constraint Satisfaction Problem implementation. This problem contains an unused initial state field to accommodate the requirements of some search algorithms.

	Note: you will need to define the constraint function for each specific problem!
	"""
	struct CSP <: AbstractCSP
		vars::Vector
		domains::Dict
		neighbors::Dict
		constraints::Function
		initial::Tuple
		current_domains::Dict # used for inference
		nassigns::Vector

		function CSP(vars::Vector, domains::Dict, neighbors::Dict, constraints::Function;
					initial::Tuple=(), current_domains::Union{Nothing, Dict}=nothing, nassigns::Vector=[0])
			return new(vars, domains, neighbors, constraints, initial, current_domains, nassigns)
		end
	end

	import Base: show
	
	function show(io::IO, p::CSP)
		write(io, "CSP Problem\n")
		write(io, """- variables: $(join(["$(var)" for var in p.vars],", ") )\n""")
		write(io, """- domains:\n""")
		for var in p.vars
			write(io, """  $(rpad(var,3,' ')): $(join(p.domains[var], ", "))\n""")
		end
		write(io,"- number of assignments: $(p.nassigns[1])")
	end

	nothing
end

# ╔═╡ 1cd48b57-92da-46dc-aeab-99c6712aa61a
md"""
### General backtracking algorithm
"""

# ╔═╡ af6a7d8c-66cc-4230-8ef2-1b4fbb461d9a
md"""
### Variable selection methods
"""

# ╔═╡ 6aef41ad-ac04-4974-92b8-fce053eb5a08
md"""
### Value selection methods
"""

# ╔═╡ df53d869-90ac-4397-96dd-370d36da381a
md"""
### Inference methods
"""

# ╔═╡ deee7ef8-916b-4496-8ef3-20562cb79606
begin
	"""
		support_pruning(csp::CSP)

	set prunable domains dictionaries
	"""
	function support_pruning(csp::CSP)
		if iszero(length(csp.current_domains))
			for var in csp.vars
				csp.current_domains[var] = copy(csp.domains[var]) # requires copy for this to work (without a copy, pruning can generate unwanted results in the case where the domain of each variable refers to the same object)
			end
		end
		return
	end

	"""
		suppose(csp::CSP, var, val)

	accumulate removed values for inference assuming var=value. returns list of removed values for a variable
	"""
	function suppose(csp::CSP, var, val)
		support_pruning(csp)
		removals = [Pair(var, a) for a in csp.current_domains[var] if (a != val)]
		csp.current_domains[var] = [val]
		return removals
	end

	"""
		restore(csp::CSP, removals)

	restore values that were removed from a domain to run inference
	"""
	function restore(csp::CSP, removals)
		for (var, val) in removals
			push!(csp.current_domains[var], val);
		end
		return
	end

	"""
		no_inference(csp::CSP, var, value, assignment::Dict, removals)

	Inference method that does nothing
	"""
	function no_inference(csp::CSP, var, value, assignment::Dict, removals)
		return true
	end


	"""
		forward_checking(csp::CSP, var, value, assignment::Dict, removals)

	Forward checking implementation
	"""
	function forward_checking(csp::CSP, var, value, assignment::Dict, removals)
		#@debug "running my inference for:\n $(var) with value $(value)\n current assignment: $(assignment)\n current removals: $(removals)"
		for B in csp.neighbors[var]
			if !haskey(assignment, B)
				@debug "working on $(B)"
				# remove the conflicting values
				for b in csp.current_domains[B]
					if !csp.constraints(var, value, B, b)
						@debug "assignment $(var):$(value) conflicts with $(B):$(b)"
						prune(csp, B, b, removals)
					end
				end
				# check if any possible values left
				if length(csp.current_domains[B]) == 0
					return false
				end
			end
		end
		return true
	end

	"""
		prune(csp::CSP, var, value, removals)

	Helper function for `forward_checking`. Does the actual removal
	"""
	function prune(csp::CSP, var, value, removals)
		# find index of value if it exists
		ind = findfirst(x->isequal(x, value), csp.current_domains[var])
		# remove value for variable
		if !isnothing(ind) 
			@debug "removing value $(value) for variable $(var)"
			deleteat!(csp.current_domains[var], ind)
			push!(removals, Pair(var, value))
		end

		nothing
	end

	nothing
end

# ╔═╡ 49f0a27f-a004-4f3c-a182-0a2b98fe791b
begin
	"""
		backtrack(csp)
	
	Search the given problem by using the backtracking search algorithm (Fig 5.5) and return the solution
	if any are found.

	kwargs
	------
		- select_unassigned_variable : Function
			by default returns a list of unasigned variables
		- order_domain_values : Function
			by default returns a list of possible domain value (no specific order)
		- inference : Function
			defaults to something that does nothing. Can be something else, e.g. forward_checking 

	"""
	function backtrack(csp::CSP, 
					   assignment::Dict=Dict{eltype(csp.vars), eltype(csp.domains[csp.vars[1]])}();
					   select_unassigned_variable::Function=first_unassigned_variable,
					   order_domain_values::Function=unordered_domain_values,
					   inference::Function=no_inference, level=1)
		# goal test
		if length(assignment) == length(csp.vars)
			if goal_test(csp, assignment)
				return assignment
			else
				error("BacktrackingSearchError: search failed to find a solution!")
			end
		end

		# recursive work
		var = select_unassigned_variable(csp, assignment)		   			# for unassigned variables
		values = order_domain_values(csp, var, assignment) 					# for each value
		@debug "\n- current level: $(level)\n- current variable: $(var)\n- assignment:\n $(assignment)\n- current domains:\n  $(csp.current_domains)\npossible values: $(values)"
		for value in values
			if nconflicts(csp, var, value, assignment) == 0					# if value is consistent
				assign(csp, var, value, assignment) 						# assign value
				removals = suppose(csp, var, value)  						# determing removals for inference	
				@debug "\ncurrent removals: $(removals)"
				if inference(csp, var, value, assignment, removals)			# run inference if required
					result = backtrack(csp, assignment; select_unassigned_variable=select_unassigned_variable,
																order_domain_values=order_domain_values,
																inference=inference,
																level=level+1)
					if !isnothing(result)
						return result 
					end
				end
				restore(csp, removals)  									# undo inferences if required
			end
		end
		unassign(csp, var, assignment)  									# undo variable assignment
		
		return nothing;
	end

	"""
		goal_test(csp::CSP, assignment)
		
	Check if a goal state has been reached given an assignment
	"""
	function goal_test(csp::CSP, assignment)
		return (length(assignment) == length(csp.vars)) && all(x->nconflicts(csp, x, assignment[x], assignment) == 0, csp.vars)
	end
	

	"""
		assign(csp, var, val, assignment)
	
	Overwrite (if an value exists already) assignment[key] with 'val'.
	"""
	function assign(csp::CSP, var, value, assigment::Dict)
		assigment[var] = value
		csp.nassigns .+= 1
		return
	end

	"""
		unassign(csp, var, val, assignment)
	
	Delete the existing (var, val) pair from 'assignment'.
	"""
	function unassign(csp::CSP, var, assignment::Dict)
		delete!(assignment, var)
	end

	"""
		nconflicts(csp, var, val, assignment)

	Count the number of conflicts given the assignment of var to val.
	"""
	function nconflicts(csp::CSP, var, val, assignment::Dict)
		res = count(x -> haskey(assignment, x) && !csp.constraints(var, val, x, assignment[x]), csp.neighbors[var] )
		@debug """Call to nconflict for variable $(var) with value $(val) found $(res) conflicts"""
		return res
	end

	nothing
end

# ╔═╡ 167bbc38-43dc-476e-aa51-3735a783202d
begin
	"""
		first_unassigned_variable(csp::CSP, assignment::Dict)
	
	Variable selection method that picks the first variable who has not been assigned yet.
	"""
	function first_unassigned_variable(csp::CSP, assignment::Dict)
		return csp.vars[findfirst(x-> !haskey(assignment, x), csp.vars)]
	end

	"""
		minimum_remaining_values(csp::CSP, assignment::Dict)

	Variable selection method that picks the variable with the least amount of remaining values.
	"""
	function minimum_remaining_values(csp::CSP, assignment::Dict)
		return argmin(x -> num_legal_values(csp, x, assignment), [var for var in csp.vars if !haskey(assignment, var)])
	end

	"""
		num_legal_values(csp::CSP, var, assignment::Dict)

	Helper function for `minimum_remaining_values`.
	"""
	function num_legal_values(csp::CSP, var, assignment::Dict)
	    if haskey(csp.current_domains, var)
	        return length(csp.current_domains[var])
	    else
	        return count(val -> nconflicts(csp, var, val, assignment) == 0, csp.domains[var])
		end
	end

	nothing
end

# ╔═╡ 0e5f408a-0c28-4561-99a6-b9f5088f612d
begin	
	"""
		unordered_domain_values(csp::CSP, var, assignment::Dict)

	Value selection method that picks the available values in no specific order.
	"""
	function unordered_domain_values(csp::CSP, var, assignment::Dict)
		return choices(csp, var)
	end

	"""
		choices(csp::CSP, var)

	Helper function for variable selection (used by `unordered_domain_values`)
	"""
	function choices(csp::CSP, var)
	    if haskey(csp.current_domains, var)
	        return csp.current_domains[var]
	    else
	        return csp.domains[var]
	    end
	end

	"""
		least_constraining_values(csp::CSP, var, assignment::Dict) 

	Value selection method that return the available values sorted by the number of conflicts (least constrained first)
	"""
	function least_constraining_values(csp::CSP, var, assignment::Dict) 
    	return sort(choices(csp, var), by=val->nconflicts(csp, var, val, assignment))
	end

	nothing
end

# ╔═╡ fc61e9b9-319d-4813-a62d-c5bcd518eb6a
md"""
## Applications 
### Graph coloring (bis)
We can now build the setup for the graph coloring problem. We will need to define the following items:
* constraint checker
* a function to generate a CSP from the problem description
* a function for variable ordering (as it is imposed)
* a function for value sorting (imposed as well)

You can compare the solutions using the default settings and the value and variable ordering functions that are imposed:

$(PlutoUI.LocalResource("./img/csp_support_compare.png", :width => 420, :align => "center"))

"""

# ╔═╡ 6f45541b-d049-491a-9763-82f9138f3824
begin
	"""
		colorconstraints()

	Helper function to check if an assignment is compatible with the constraints. Returns true if ok.
	"""
	function colorconstraints(var_1, val_1, var_2, val_2)
		return !isequal(val_1, val_2)
	end
	
	"""
		colorproblem()

	Generate CSP for the colorproblem
	"""
	function colorproblem()
		vars = [:A1;:A2;:A3;:A4;:H;:T;:F1;:F2]
		dom = [:red; :green; :blue]
		domains = Dict(var => dom for var in vars)
		neighbors = Dict(:A1 => [:A2; :H],
						 :A2 => [:A1; :A3; :H],
						 :A3 => [:A2; :A4; :H],
						 :A4 => [:A3; :H],
						 :H  => [:A1; :A2; :A3; :A4; :T],
						 :T  => [:H; :F1; :F2],
						 :F1 => [:T],
						 :F2 => [:T])
		
		return CSP(vars, domains, neighbors, colorconstraints, current_domains=Dict{eltype(vars), typeof(dom)}())
	end

	const colorvarorder = [:A1; :H; :A4; :F1; :A2; :F2; :A3; :T]
	const colorvalorder = Dict(:red => 1, :green => 2, :blue => 3)
	
	function select_unassigned_variable_inorder(csp::CSP, assignment)
		newvar = colorvarorder[findfirst(x-> !haskey(assignment, x), colorvarorder)]
		@debug "VARIABLE SELECTION\ncurrent assignment: $(assignment)\nnext variable: $(newvar)"
		return newvar
	end
	
	select_unassigned_value_inorder(csp::CSP, var, assignment) = sort(haskey(csp.current_domains,var) ? csp.current_domains[var] : csp.domains[var], by=x->colorvalorder[x])
	
	
end

# ╔═╡ 40f100b9-e893-4f37-a3af-4bcad05d5289
let
 	# using default variable and value ordering
	io = open("log_default.txt", "w+")
	Logging.disable_logging(LogLevel(-5000))
	logger = SimpleLogger(io, LogLevel(-5000))
	p = colorproblem()
	sol = nothing
	with_logger(logger) do
		@debug p.current_domains, length(p.current_domains)
		sol = backtrack(p)
	end
	flush(io)
	Logging.disable_logging(LogLevel(-1000))
	
	println("Node -> Assignment \n", join(["$(var) -> $(sol[var])" for var in colorvarorder], "\n"))
end

# ╔═╡ eb17aa51-2126-447e-8f27-4a41388f0263
let
	# using imposed variable and value ordering
	Logging.disable_logging(LogLevel(-5000))
	io = open("log_imposed.txt", "w+")
	logger = SimpleLogger(io, LogLevel(-1000))
	p = colorproblem()
	sol = nothing
	with_logger(logger) do
		@debug p.current_domains, length(p.current_domains)
		sol = backtrack(p; select_unassigned_variable=select_unassigned_variable_inorder, 
						   order_domain_values=select_unassigned_value_inorder)
	end
	flush(io)
	Logging.disable_logging(LogLevel(-1000))
	println("Node -> Assignment \n", join(["$(var) -> $(sol[var])" for var in colorvarorder], "\n"))
end

# ╔═╡ e3db6424-71d6-11eb-0b85-db4e9e6f4bcf
md"""
### Zebra puzzle
Consider the following logic puzzle: In five houses, each with a different color, live five persons of different nationalities, each of whom prefers a different brand of candy, a different drink, and a different pet. Given the following facts, the questions to answer are “Where does the zebra live, and in which house do they drink water?”
* The Englishman lives in the red house.
* The Spaniard owns the dog.
* The Norwegian lives in the first house on the left.
* The green house is immediately to the right of the ivory house.
* The man who eats Hershey bars lives in the house next to the man with the fox. 
* Kit Kats are eaten in the yellow house.
* The Norwegian lives next to the blue house.
* The Smarties eater owns snails.
* The Snickers eater drinks orange juice.
* The Ukrainian drinks tea.
* The Japanese eats Milky Ways.
* Kit Kats are eaten in a house next to the house where the horse is kept. Coffee is drunk in the green house.
* Milk is drunk in the middle house.

Questions:
1. What could be possible variables, domains and constraints?
2. Try to solve the zebra puzzle (using the methods from the course, not by hand)

"""

# ╔═╡ 01852eb5-67a6-4d42-bb89-da6133d1bd9e
begin
	const colors = ["Red";"Yellow";"Blue";"Green";"Ivory"]
	const pets = ["Dog";"Fox";"Snails";"Horse";"Zebra"]
	const drinks = ["OJ";"Tea";"Coffee";"Milk";"Water"]
	const countries = ["Englishman";"Spaniard";"Norwegian";"Ukranian";"Japanese"]
	const candies = ["KitKat";"Hershey";"Smarties";"Snickers";"Milkyway"]
end
	

# ╔═╡ 297786ea-71d5-11eb-3401-05d1c135d4c6
md"""
### Exam scheduling
Consider the exam scheduling problem. We want to schedule 5 exams in the period 1-19 June. Each exam has a minimum number of days required to be able to study for it. The faculty members have some other things to do as well, so their availability is limited. The table below gives you an overview. The assignment of a course to a given date $D$ is considered as a block that starts at $D-\text{preparation days}$. The actual exam period starts on the first, and preparation days should take place in the exam period as well (e.g. SE426 can take place NET the 5th). Exams are never scheduled on the same day.

| Course  | available dates staff  |  required preparation days |
|---|---|---|
| DS425	| $1\rightarrow19$ | 3 |
|SE422	| $2\rightarrow6$ |	2 |
|SE423	| $8\rightarrow19$ |	4 |
|SE426	| $1\rightarrow16$ |	4 |
|TN421	| $1\rightarrow19$ |	1|

1. Sketch the constraint graph for this problem.
2. Use the backtracking algorithm to solve this problem. Use the minimum remaining values heuristic for variable selection, the least constraining value for value ordering and forward checking for inference. In the case of ties, use the value/variable that occurs first using numerical/alphabetical order.

"""

# ╔═╡ d737e518-926e-439d-99f2-c9061383af00


# ╔═╡ 8a65d932-71df-11eb-0a50-a30132237e47
md"""
### Map coloring revisited
We consider the example from the lectures. The objective of this application is to compare the performance of the backtracking search using different combinations for variable selection, value selection and inference.

Implement the map coloring problem and benchmark the performance of different combinations.
"""

# ╔═╡ a467991a-3e78-4bce-b0a4-332bed05d540


# ╔═╡ Cell order:
# ╟─9236c53d-accf-44e4-9a95-f71cfda90c03
# ╟─902eeb2e-71d2-11eb-3322-53451cb23bf2
# ╟─d0fa06da-71d1-11eb-2d00-97ea31bbb5ed
# ╟─1a8ce0dd-bece-437e-b44f-a31f8c155ce1
# ╟─4b510a0c-55bb-4e28-8dcd-993ee8ef433f
# ╟─f95662cb-7034-4efd-916b-57c79cd8929c
# ╟─6727b2df-d549-459b-9332-c7b607f09df2
# ╠═82d6b21c-5a95-4cdb-bd0b-e061dec50169
# ╟─1cd48b57-92da-46dc-aeab-99c6712aa61a
# ╠═49f0a27f-a004-4f3c-a182-0a2b98fe791b
# ╟─af6a7d8c-66cc-4230-8ef2-1b4fbb461d9a
# ╠═167bbc38-43dc-476e-aa51-3735a783202d
# ╟─6aef41ad-ac04-4974-92b8-fce053eb5a08
# ╠═0e5f408a-0c28-4561-99a6-b9f5088f612d
# ╟─df53d869-90ac-4397-96dd-370d36da381a
# ╠═deee7ef8-916b-4496-8ef3-20562cb79606
# ╟─fc61e9b9-319d-4813-a62d-c5bcd518eb6a
# ╠═6f45541b-d049-491a-9763-82f9138f3824
# ╠═40f100b9-e893-4f37-a3af-4bcad05d5289
# ╠═eb17aa51-2126-447e-8f27-4a41388f0263
# ╟─e3db6424-71d6-11eb-0b85-db4e9e6f4bcf
# ╠═01852eb5-67a6-4d42-bb89-da6133d1bd9e
# ╟─297786ea-71d5-11eb-3401-05d1c135d4c6
# ╠═d737e518-926e-439d-99f2-c9061383af00
# ╟─8a65d932-71df-11eb-0a50-a30132237e47
# ╠═a467991a-3e78-4bce-b0a4-332bed05d540
