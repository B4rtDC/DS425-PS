### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 902eeb2e-71d2-11eb-3322-53451cb23bf2
# dependencies
begin
	using PlutoUI
	using Logging
	Logging.global_logger(Logging.SimpleLogger(stdout, LogLevel(-5000)  ) )
	Logging.disable_logging(LogLevel(-3001))
end

# ╔═╡ d0fa06da-71d1-11eb-2d00-97ea31bbb5ed
md"""# Constraint Satisfaction Problems (CSP)"""

# ╔═╡ 7321b9ee-71d2-11eb-1338-b35f914af6bf
md"""
## Application 1
How many solutions are there for the map-coloring problem in Figure 6.1 using RGB? How many solutions if four colors are allowed? Two colors? What about $n$ colors?


$(PlutoUI.LocalResource("./img/map.png", :width => 500, :align => "middle"))



"""

# ╔═╡ 13e93050-71d3-11eb-14a9-c145d4e13c8e
md"""
## Application 2
Give precise formulations for each of the following as constraint satisfaction problems:
* Rectilinear floor-planning:

    The problem of allocating given rooms within a given predefined contour shape, while satisfying given adjacency and size constraints. This work is significant with respect to complex building structures, such as hospitals, schools, offices, etc. You need to find non-overlapping places in a large rectangle for a number of smaller rectangles.

$(PlutoUI.LocalResource("./img/floorplan.gif", "width" => 500))
    

* Hamiltonian tour: given a network of cities connected by roads, choose an order to visit all cities in a country without repeating any. You can consider this as a relaxed version of the TSP, as the "cost" of the entire trip is not taken into account.
"""

# ╔═╡ 297786ea-71d5-11eb-3401-05d1c135d4c6
md"""
## Application 3
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

# ╔═╡ e3db6424-71d6-11eb-0b85-db4e9e6f4bcf
md"""
## Application 4  (Limited coding)
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
2. Try to understand the implementation that is given below and come up with a solution. Hint: rubber ducking it might be a good way of doing this.

"""

# ╔═╡ 6f572702-71d9-11eb-2871-d707f3ee1df7
begin
	
	# to be able to add new methods to the existing ones in Julia Base
	import Base.getindex, Base.get, Base.haskey, Base.in, Base.copy, Base.deepcopy
	
	abstract type AbstractProblem end;
	abstract type AbstractCSP <: AbstractProblem end;
	
	# ___________________________________________________________________ #
	# Everything related to dictionaries for domain and neighbor tracking #
	# ___________________________________________________________________ #
	struct ConstantFunctionDict{V}
		value::V
		function ConstantFunctionDict{V}(val::V) where V
			return new(val);
		end
	end

	ConstantFunctionDict(val) = ConstantFunctionDict{typeof(val)}(val);

	copy(cfd::ConstantFunctionDict) = ConstantFunctionDict{typeof(cfd.value)}(cfd.value);

	deepcopy(cfd::ConstantFunctionDict) = ConstantFunctionDict{typeof(cfd.value)}(deepcopy(cfd.value));

	mutable struct CSPDict
		dict::Union{Nothing, Dict, ConstantFunctionDict}

		function CSPDict(dictionary::Union{Nothing, Dict, ConstantFunctionDict})
			return new(dictionary);
		end
	end

	function getindex(dict::CSPDict, key)
		if (typeof(dict.dict) <: ConstantFunctionDict)
			return dict.dict.value;
		else
			return getindex(dict.dict, key);
		end
	end

	function getkey(dict::CSPDict, key, default)
		if (typeof(dict.dict) <: ConstantFunctionDict)
			return dict.dict.value;
		else
			return getkey(dict.dict, key, default);
		end
	end

	function get(dict::CSPDict, key, default)
		if (typeof(dict.dict) <: ConstantFunctionDict)
			return dict.dict.value;
		else
			return get(dict.dict, key, default);
		end
	end

	function haskey(dict::CSPDict, key)
		if (typeof(dict.dict) <: ConstantFunctionDict)
			return true;
		else
			return haskey(dict.dict, key);
		end
	end

	function in(pair::Pair, dict::CSPDict)
		if (typeof(dict.dict) <: ConstantFunctionDict)
			if (getindex(pair, 2) == dict.dict.value)
				return true;
			else
				return false;
			end
		else
			return in(pair, dict.dict);
		end
	end

	# ___________________________________________________________________ #
	# Everything related to assigning variables #
	# ___________________________________________________________________ #

	"""
		assign(problem, key, val, assignment)
	Overwrite (if an value exists already) assignment[key] with 'val'.
	"""
	function assign(problem::T, key, val, assignment::Dict) where {T <: AbstractCSP}
		assignment[key] = val;
		problem.nassigns = problem.nassigns + 1;
		nothing;
	end

	"""
		unassign(problem, key, val, assignment)
	Delete the existing (key, val) pair from 'assignment'.
	"""
	function unassign(problem::T, key, assignment::Dict) where {T <: AbstractCSP}
		if (haskey(assignment, key))
			delete!(assignment, key);
		end
		nothing;
	end

	function nconflicts(problem::T, key, val, assignment::Dict) where {T <: AbstractCSP}
		return count(
					(function(second_key)
						return (haskey(assignment, second_key) &&
							!(problem.constraints(key, val, second_key, assignment[second_key])));
					end),
					problem.neighbors[key]);
	end


	function support_pruning(problem::T) where {T <: AbstractCSP}
		if (problem.current_domains === nothing)
			problem.current_domains = Dict(collect(Pair(key, collect(problem.domains[key])) for key in problem.vars));
		end
		nothing;
	end

	function suppose(problem::T, key, val) where {T <: AbstractCSP}
		support_pruning(problem);
		local removals::AbstractVector = collect(Pair(key, a) for a in problem.current_domains[key]
												if (a != val));
		problem.current_domains[key] = [val];
		return removals;
	end


	function parse_neighbors(neighbors::String; vars::AbstractVector=[])
		local new_dict = Dict();
		for var in vars
			new_dict[var] = [];
		end
		local specs::AbstractVector = collect(map(String, split(spec, [':'])) for spec in split(neighbors, [';']));
		for (A, A_n) in specs
			A = String(strip(A));
			if (!haskey(new_dict, A))
				new_dict[A] = [];
			end
			for B in map(String, split(A_n))
				push!(new_dict[A], B);
				if (!haskey(new_dict, B))
					new_dict[B] = [];
				end
				push!(new_dict[B], A);
			end
		end
		return new_dict;
	end
	
	
	"""
		CSP
	
	CSP is a Constraint Satisfaction Problem implementation of AbstractProblem and AbstractCSP. This problem contains an unused initial state field to accommodate the requirements of some search algorithms.

	Note: you will need to define the constraint function for each specific problem! cf.  zebra_constraint, different_values_constraint etc.
	
	"""
	mutable struct CSP <: AbstractCSP
		vars::AbstractVector
		domains::CSPDict
		neighbors::CSPDict
		constraints::Function
		initial::Tuple
		current_domains::Union{Nothing, Dict}
		nassigns::Int64

		function CSP(vars::AbstractVector, domains::CSPDict, neighbors::CSPDict, constraints::Function;
					initial::Tuple=(), current_domains::Union{Nothing, Dict}=nothing, nassigns::Int64=0)
			return new(vars, domains, neighbors, constraints, initial, current_domains, nassigns)
		end
	end
	
	# Initialising the problem's constants
	struct ZebraInitialState
		colors::Array{String, 1}
		pets::Array{String, 1}
		drinks::Array{String, 1}
		countries::Array{String, 1}
		candies::Array{String, 1}

		function ZebraInitialState()
			local colors = map(String, split("Red Yellow Blue Green Ivory"));
			local pets = map(String, split("Dog Fox Snails Horse Zebra"));
			local drinks = map(String, split("OJ Tea Coffee Milk Water"));
			local countries = map(String, split("Englishman Spaniard Norwegian Ukranian Japanese"));
			local candies = map(String, split("KitKat Hershey Smarties Snickers Milkyway"));
			return new(colors, pets, drinks, countries, candies);
		end
	end

	zebra_constants = ZebraInitialState();
	
	"""
		ZebraCSP
	
	ZebraCSP is a Zebra Constraint Satisfaction Problem implementation of AbstractProblem and AbstractCSP.
	"""
	mutable struct ZebraCSP <: AbstractCSP
		vars::AbstractVector
		domains::CSPDict
		neighbors::CSPDict
		constraints::Function
		initial::Tuple
		current_domains::Union{Nothing, Dict}
		nassigns::Int64

		function ZebraCSP(;initial::Tuple=(), current_domains::Union{Nothing, Dict}=nothing, nassigns::Int64=0)
			# 25 variables associated with the problem
			local vars = vcat(zebra_constants.colors,
							zebra_constants.pets,
							zebra_constants.drinks,
							zebra_constants.countries,
							zebra_constants.candies);
			local domains = Dict();
			# Set all possible domain values equal to 1,2, ... , 5
			for var in vars
				domains[var] = collect(1:5);
			end
			# impose constraints
			domains["Norwegian"] = [1];
			domains["Milk"] = [3];
			neighbors = parse_neighbors("Englishman: Red;
					Spaniard: Dog; KitKat: Yellow; Hershey: Fox;
					Norwegian: Blue; Smarties: Snails; Snickers: OJ;
					Ukranian: Tea; Japanese: Milkyway; KitKat: Horse;
					Coffee: Green; Green: Ivory", vars=vars);
			for category in [zebra_constants.colors,
							zebra_constants.pets,
							zebra_constants.drinks,
							zebra_constants.countries,
							zebra_constants.candies]
				for A in category
					for B in category
						if (A != B)
							if (!(B in neighbors[A]))
								push!(neighbors[A], B);
							end
							if (!(A in neighbors[B]))
								push!(neighbors[B], A);
							end
						end
					end
				end
			end
			# return correctyl defined problem
			return new(vars, CSPDict(domains), CSPDict(neighbors), zebra_constraint, initial, current_domains, nassigns);
		end
	end
	
	

	# deal with the constraints
	function zebra_constraint(A::String, a, B::String, b; recursed::Bool=false)
		local same::Bool = (a == b); # identical numerical value
		local next_to::Bool = (abs(a - b) == 1); # neigbors
		if (A == "Englishman" && B == "Red")
			return same;
		elseif (A == "Spaniard" && B == "Dog")
			return same;
		elseif (A == "Hershey" && B == "Fox")
			return next_to;
		elseif (A == "Norwegian" && B == "Blue")
			return next_to;
		elseif (A == "KitKat" && B == "Yellow")
			return same;
		elseif (A == "Smarties" && B == "Snails")
			return same;
		elseif (A == "Snickers" && B == "OJ")
			return same;
		elseif (A == "Ukranian" && B == "Tea")
			return same;
		elseif (A == "Japanese" && B == "Milkyway")
			return same;
		elseif (A == "KitKat" && B == "Horse")
			return next_to;
		elseif (A == "Coffee" && B == "Green")
			return same;
		elseif (A == "Green" && B == "Ivory")
			return ((a - 1) == b);
		elseif (!recursed)
			return zebra_constraint(B, b, A, a, recursed=true);
		elseif ((A in zebra_constants.colors && B in zebra_constants.colors) ||
				(A in zebra_constants.pets && B in zebra_constants.pets) ||
				(A in zebra_constants.drinks && B in zebra_constants.drinks) ||
				(A in zebra_constants.countries && B in zebra_constants.countries) ||
				(A in zebra_constants.candies && B in zebra_constants.candies))
			return !same;
		else
			error("ZebraConstraintError: This constraint could not be evaluated on the given arguments!");
		end
	end

	

	function solve_zebra(problem::ZebraCSP, algorithm::Function; kwargs...)
		local answer = algorithm(problem; kwargs...);
		for house in collect(1:5)
			print("House ", house);
			for (key, val) in collect(answer)
				if (val == house)
					print(" ", key);
				end
			end
			println();
		end
		return answer["Zebra"], answer["Water"], problem.nassigns, answer;
	end
end;

# ╔═╡ adc27b3a-71da-11eb-325e-011cf335877e
begin
	# ___________________________________________________________________ #
	#               Actual backtracking search algorithm                  #
	# ___________________________________________________________________ #

	"""
		backtracking_search(problem)
	Search the given problem by using the backtracking search algorithm (Fig 6.5) and return the solution
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
	function backtracking_search(problem::T;
								select_unassigned_variable::Function=first_unassigned_variable,
								order_domain_values::Function=unordered_domain_values,
								inference::Function=no_inference) where {T <: AbstractCSP}
		local result = backtrack(problem, Dict(),
								select_unassigned_variable=select_unassigned_variable,
										order_domain_values=order_domain_values,
										inference=inference);
		if (!(typeof(result) <: Nothing || goal_test(problem, result)))
			error("BacktrackingSearchError: Unexpected result!")
		end
		return result;
	end

	function backtrack(problem::T, assignment::Dict;
						select_unassigned_variable::Function=first_unassigned_variable,
						order_domain_values::Function=unordered_domain_values,
						inference::Function=no_inference) where {T <: AbstractCSP}
		if (length(assignment) == length(problem.vars))
			return assignment;
		end
		local var = select_unassigned_variable(problem, assignment);
		for value in order_domain_values(problem, var, assignment)
			if (nconflicts(problem, var, value, assignment) == 0)
				assign(problem, var, value, assignment);
				removals = suppose(problem, var, value);
				if (inference(problem, var, value, assignment, removals))
					result = backtrack(problem, assignment,
										select_unassigned_variable=select_unassigned_variable,
										order_domain_values=order_domain_values,
										inference=inference);
					if (!(typeof(result) <: Nothing))
						return result;
					end
				end
				restore(problem, removals);
			end
		end
		unassign(problem, var, assignment);
		return nothing;
	end


	function restore(problem::T, removals::AbstractVector) where {T <: AbstractCSP}
		for (key, val) in removals
			push!(problem.current_domains[key], val);
		end
		nothing;
	end

	function goal_test(problem::T, state::Tuple) where {T <: AbstractCSP}
		let
			local assignment = Dict(state);
			return (length(assignment) == length(problem.vars) &&
					all((function(key)
								return nconflicts(problem, key, assignment[key], assignment) == 0;
							end)
							,
							problem.vars));
		end
	end

	function goal_test(problem::T, state::Dict) where {T <: AbstractCSP}
		let
			local assignment = deepcopy(state);
			return (length(assignment) == length(problem.vars) &&
					all((function(key)
								return nconflicts(problem, key, assignment[key], assignment) == 0;
							end),
							problem.vars));
		end
	end


	function first_unassigned_variable(problem::T, assignment::Dict) where {T <: AbstractCSP}
		return getindex(problem.vars, findfirst((function(var)
							return !haskey(assignment, var);
						end),
						problem.vars));
	end

	function unordered_domain_values(problem::T, var, assignment::Dict) where {T <: AbstractCSP}
		return choices(problem, var);
	end

	function choices(problem::T, key) where {T <: AbstractCSP}
		if (!(problem.current_domains === nothing))
			return problem.current_domains[key];
		else
			return problem.domains[key];
		end
	end

	function no_inference(problem::T, var, value, assignment::Dict, removals::Union{Nothing, AbstractVector}) where {T <: AbstractCSP}
		return true;
	end
	
	# ___________________________________________________________________ #
	#  inference option (forward checking, not used standardbacktracking) #
	# ___________________________________________________________________ #

	function forward_checking(problem::T, var, value, assignment::Dict, removals::Union{Nothing, AbstractVector}) where {T <: AbstractCSP}
		for B in problem.neighbors[var]
			if (!haskey(assignment, B))
				for b in copy(problem.current_domains[B])
					if (!problem.constraints(var, value, B, b))
						prune(problem, B, b, removals);
					end
				end
				if (length(problem.current_domains[B]) == 0)
					return false;
				end
			end
		end
		return true;
	end

	function prune(problem::T, key, value, removals) where {T <: AbstractCSP}
		local not_removed::Bool = true;
		for (i, element) in enumerate(problem.current_domains[key])
			if (element == value)
				deleteat!(problem.current_domains[key], i);
				not_removed = false;
				break;
			end
		end
		if (not_removed)
			error("Could not find ", value, " in ", problem.current_domains[key], " for key '", key, "' to be removed!");
		end
		if (!(typeof(removals) <: Nothing))
			push!(removals, Pair(key, value));
		end
		nothing;
	end
end;

# ╔═╡ be0f324e-71da-11eb-12da-43a7c3b865ec
begin
	prob = ZebraCSP()
	# Solve zebra problem with backtracking search
	@time solve_zebra(prob, backtracking_search);
end

# ╔═╡ 8a65d932-71df-11eb-0a50-a30132237e47
md"""
## Application 5 - another look at map coloring
"""

# ╔═╡ 9b4a7ab6-71df-11eb-2324-b115dd57c628
begin
	# function to identify if two values are the same (in case of node coloring)
	function different_values_constraint(A::T1, a::T2, B::T1, b::T2) where {T1, T2}
		return (a != b);
	end

	function MapColoringCSP(colors::AbstractVector, neighbors::String)
		local parsed_neighbors = parse_neighbors(neighbors);
		return CSP(collect(keys(parsed_neighbors)), CSPDict(ConstantFunctionDict(colors)), CSPDict(parsed_neighbors), different_values_constraint);
	end

	function MapColoringCSP(colors::AbstractVector, neighbors::Dict)
		return CSP(collect(keys(neighbors)), CSPDict(ConstantFunctionDict(colors)), CSPDict(neighbors), different_values_constraint);
	end
	
end;

# ╔═╡ b9bafd2c-71df-11eb-05ee-55793f3a78fa
begin
	@info "Comparing performance for map coloring"
	
	australia_csp = MapColoringCSP(["R", "G", "B"], "SA: WA NT Q NSW V; NT: WA Q; NSW: Q V; T: ");
	@time backtracking_search(australia_csp)
	@info *("Backtracking search without inference, $(australia_csp.nassigns) variables assigned\n", ["\t$(key): $(val)\n" for (key, val) in australia_csp.current_domains]...)
	
	australia_csp = MapColoringCSP(["R", "G", "B"], "SA: WA NT Q NSW V; NT: WA Q; NSW: Q V; T: ");
	@time backtracking_search(australia_csp; inference=forward_checking)
	@info *("Backtracking search with inference (forward checking), $(australia_csp.nassigns) variables assigned\n", ["\t$(key): $(val)\n" for (key, val) in australia_csp.current_domains]...)
end

# ╔═╡ 4f044672-7fe1-11eb-2a21-dbd705daf5e8
australia_csp.vars

# ╔═╡ Cell order:
# ╠═902eeb2e-71d2-11eb-3322-53451cb23bf2
# ╟─d0fa06da-71d1-11eb-2d00-97ea31bbb5ed
# ╟─7321b9ee-71d2-11eb-1338-b35f914af6bf
# ╟─13e93050-71d3-11eb-14a9-c145d4e13c8e
# ╟─297786ea-71d5-11eb-3401-05d1c135d4c6
# ╟─e3db6424-71d6-11eb-0b85-db4e9e6f4bcf
# ╠═6f572702-71d9-11eb-2871-d707f3ee1df7
# ╠═adc27b3a-71da-11eb-325e-011cf335877e
# ╠═be0f324e-71da-11eb-12da-43a7c3b865ec
# ╠═8a65d932-71df-11eb-0a50-a30132237e47
# ╠═9b4a7ab6-71df-11eb-2324-b115dd57c628
# ╠═b9bafd2c-71df-11eb-05ee-55793f3a78fa
# ╠═4f044672-7fe1-11eb-2a21-dbd705daf5e8
