### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ bca1c92c-8561-11eb-0420-6f0123bacb7f
begin
	# dependencies
	
	using Pkg
	Pkg.activate(joinpath(@__DIR__,".."))
	
	using PlutoUI
	#using DataStructures
	#using Logging
	#Logging.global_logger(Logging.SimpleLogger(stdout, LogLevel(-5000)  ) )
	nothing
	using BenchmarkTools
	using Plots
	using Graphs, SimpleWeightedGraphs, LinearAlgebra, Random
	import StatsBase:sample
	TableOfContents()
end

# ╔═╡ 8a4c3b83-4b66-41d6-a075-be34c74b456f
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

# ╔═╡ 774fa124-8560-11eb-2317-53b0d106b67a
md"""
# Local Search
Local search methods are useful when you seek a goal state and you don't really care how you got there. Examples include:
* map coloring
* resource allocation
* finding shortest routes
* ``\dots``
"""

# ╔═╡ eb014126-bd42-4581-9cf8-324dc093f964
md"""
## Quick questions
* Define local search in your own words. How does it differ from the search methods we discussed before?
* What is hill climbing (or its inverse gradient descent). What are its downsides?
* What is tabu search?
* What is simulated annealing? How does it differ from hill climbing?
* What is local beam search? How does it differ from running random restarts in parallel?
* What is a genetic (evolutionary) algorithm? What are its components?
"""

# ╔═╡ 87a06bd6-8f60-4bc8-8eee-d52f1636b56f
md"""
## Small applications
### Simulated annealing
The overview of the algorithm is shown below

  $(PlutoUI.LocalResource("./img/SA_algo.png", :width => 600, :align => "center"))

There are several aspects to consider. 
* What is considered a neighbor/successor? And how to chose them?
* How to evaluate the energy (i.e. the fitness)?
* How to let the temperature evolve. Several options exist here:
    - exponential decrease: $T = T_0(1-\alpha)^k$, where $\alpha$ is the ‘cooling rate’ and $k$ is the iteration
    - fast decrease: $T = T_0/k$
    - boltzmann decrease: $T = T_0/log(k)$  


To illustrate the principle, let's try to find the maximum of 
`` f(x) = -\dfrac{1}{4}x^4 +  \dfrac{7}{3} x^3 - 7x^2 + 8x `` in the interval $[0, 5]$. 
"""

# ╔═╡ 9db116bd-7ddc-42ef-ba26-b9538a9b89c1
f(x) = -1/4*x.^4 + 7/3 *x.^3 - 7*x.^2 + 8*x

# ╔═╡ d47e035d-c756-4510-aa10-328cbdf2f8a6
md"""
You can observe that there is a global maximum in $x=4$, however, there is a local maximum in $x=1$.

$(plot(range(0,5, length=100),f.(range(0,5, length=100)),label="",size=(300,200)))

The functions required to solve this specific problem are shown below.
"""

# ╔═╡ d5019d8a-8561-11eb-2a43-a9ce616532f5
begin
	"""
		fitness(x::Float64,f::Function)
	
	Evaluate the fitness of a specific point
	"""
	function fitness(x::Float64,f::Function)
		return f(x)
	end

	"""
		successors(x::Float64; step=0.1)
	
	Determine successors for a given location (with a fixed step size)
	You could include automatic step size modification if you want.
	"""
	function successors(x::Float64; step=0.1)
		return x .+ step*[-1; 1]
	end

	"""
		scheduletemp(t::Int64;T0::Float64=273.0, α::Float64=0.005)
	
	mapping of time to temperature. You might make this a wrapper function for
	different cooling schemes. Currently only has exponential cooling.
	"""
	function scheduletemp(t::Int64;T0::Float64=273.0, α::Float64=0.005)
		return T0*(1-α)^t
	end

	"""
		simulated_annealing(args; kwargs)
	
	Simulated annealing implementation for a 1D problem.

	Since this is stochastic (sort of a random walk), it will not always reach the optimal value, therefore, one typically runs several iterations and the best one is retained.
	
	The probability of accepting a value that leads to a decrease in energy is given by exp(ΔE/T), which is between zero (for T=0) and one.

	In the literature you might also find  1/(1 + exp(ΔE/T)) ∈ [1/2, 1] as a treshold.
	
	arguments:
	- x0: starting point
	keywords:
	- tmax: maximum iterations
	- T0: initial temperature
	- α: cooling rate
	- fitness: fitness function
	- optimfun: the function we want to optimize
	"""
	function simulated_annealing(x0; tmax=1000, T0::Float64=273.0, α::Float64=0.005,
												fitness::Function=fitness, optimfun::Function=f) 
		current = x0
		next = x0
		T = T0
		best = x0
		for t = 1:tmax
			T = scheduletemp(t,T0=T0,α=α)
			if T == 0
				return current
			else
				next = rand(successors(current))
				ΔE = fitness(next, optimfun) - fitness(current, optimfun)
				if ΔE > 0 
					current = next
				else
					if rand() < exp(ΔE/T)
						current = next
					end
				end
				if fitness(current, optimfun) > fitness(best, optimfun)
					best = current
				end
			end
		end
		return best
	end
	
	"""
		SAsearch
	
	Wrapper function for multiple SA iterations. Returns the n best found value for N runs of simulated annealing.
	"""
	function SAsearch(x0, n::Int64=1, N::Int64=10;kwargs...)
		return sort!([simulated_annealing(x0; kwargs...) for _ in 1:N], 
					 by=x->kwargs[:fitness](x,kwargs[:optimfun]),
					 rev=true)[1:n]
	end

	nothing
end

# ╔═╡ 92d33c5f-546b-4667-803d-773844a9fe7e
md"""
### Genetic algorithm
In order to implement this we will need the following items:
* an initial population (e.g. random)
* a fitness function that allows you to rank the population members
* a selection method (e.g. [roulette based proportions](https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)#Roulette_Wheel_Selection), [elite selection](https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)#Elitism_Selection))
* a [cross-over method](https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)) (cutting; 50/50 per gene etc.)
* mutation rate and method
* ``\dots``

#### Application: creating a message
Let's try write a message by using a genetic algorithm. E.g "174 Pol is better!" starting from a random set of characters (a-zA-Z0-9. !)
"""

# ╔═╡ e9cc9a3b-4353-45da-991b-2045d7aef756
begin
	# list of characters by number
	chars = vcat(65:90,97:122, 48:57,[Int(' ');Int('.'); Int('!')])
	# list of "genes"
	const genes = [Char(i) for i in chars]
	
	const pop_size = 100
	const goal = collect("174 Pol is better!");

	"""
		fitness_message(m)

	Function to evaluate the fitness of a message
	"""
	function fitness_message(m::Vector{Char}; goal=goal)
		sum(m.==goal)
	end
	nothing
end

# ╔═╡ 777949bf-590e-4f03-93e4-099f28273bbd
begin
	testcase = rand(genes, length(goal))

	println(testcase,"\n", goal, "\nTotal sequence fitness: $(fitness_message(testcase))", )
end

# ╔═╡ 6ec4f09d-6179-4636-9e16-a7a6aff9fcef
begin
	"""
		mate(p1,p2; pmut, genes)
	
	mating function to obtain children. includes both crossover and mutation.
	"""
	function mate(p1::Array{Char,1}, p2::Array{Char,1}; pmut=0.1, genes=genes)
		child = typeof(p1)() # empty child
		# walk over genome sequence 
		for i in 1:length(p1)
			p = rand()
			if p < (1-pmut)/2
				# parent 1 wins
				push!(child, p1[i])
			elseif p < 1-pmut
				# parent 2 wins
				push!(child, p2[i])
			else
				# mutation
				push!(child, rand(genes))
			end
		end
		return child
	end
	nothing
end

# ╔═╡ 9eabd1b1-e162-4194-bc96-c2d419a76b79
md"""
We can look at the distribution of the number of iterations required to solve the problem
"""

# ╔═╡ 075bebb8-73a8-4a89-a088-3a23535be67d
md"""
#### Crossover functions
Different crossover functions exist. Depending on the application you are working with, some might be more suited than others. Below you can find some algorithms. Both are described and an example as well as an implementation are provided. You are also invited to come up with a method yourself if you think you have a good idea.

##### The partially mapped crossover 
After choosing two random cut points on parents to build offspring, the portion between cut points, one parent’s string is mapped onto the other parent’s string and the remaining information is exchanged.

E.g. for parents $P_1$ and $P_2$ with cutting positions $c_1$ and $c_2$:
* ``P_1 = [3, 4, 8 | 2, 7, 1 | 6, 5 ]``
* ``P_` = [4, 2, 5 | 1, 6, 8 | 3, 7 ]``
We get the following mappings:
* ``m_{1 \rightarrow 2}: \{2 \rightarrow 1, 7 \rightarrow 6, 1 \rightarrow 8\}``
* ``m_{2 \rightarrow 1}: \{1 \rightarrow 2, 6 \rightarrow 7, 8 \rightarrow 1\}``
which leads to the following initial offspring:
* ``O_1 = [0, 0, 0 | 1, 6, 8 | 0, 0 ] ``
* ``O_2 = [0, 0, 0 | 2, 7, 1 | 0, 0 ] ``
we then add the non-conflicting values (i.e. the values not already present the offspring due do the crossover) from the original parents:
* ``O_1 = [3, 4, 0 | 1, 6, 8 | 0, 5 ]``
* ``O_2 = [4, 0, 5 | 2, 7, 1 | 3, 0 ] ``
For the remaining values, we (recursively) make use of the mappings using $m_{2 \rightarrow 1}$ for $P_1$ and $m_{1 \rightarrow 2}$ for $P_2$. This finally leads to the following offspring:
* ``O_1 = [3, 4, 2 | 1, 6, 8 | 7, 5 ] ``
* ``O_2 = [4, 8, 5 | 2, 7, 1 | 3, 6 ] ``


##### The cycle crossover 
The Cycle Crossover operator identifies a number of so-called cycles between two parent chromosomes. To form Child 1, cycle 1 is copied from parent 1, cycle 2 from parent 2, cycle 3 again from parent 1, and so on. An example might help to better understand the principle of a cycle. E.g.
* ``P_1 = [1,2,3,4,5,6,7,8] ``
* ``P_2 = [8,5,2,1,3,6,4,7] ``
For the first cycle, we start at index 1 in $P_1$. Following the links from $P_1$ to $P_2$ we find the following mappings:
* ``1 \rightarrow 8`` with index 8
* ``8 \rightarrow 7`` with index 7
* ``7 \rightarrow 4`` with index 4
* ``4 \rightarrow 1`` with index 1
after this last mapping, we are back at the starting point, so our first cycle is complete and is $\{1,4,7,8 \}$.

The second cycle will start at the first index that is not yet present in a cycle. In this case the index is equal to 2. Following the links from $P_1$ to $P_2$ we find the following mappings
* ``2 \rightarrow 5`` with index 2
* ``5 \rightarrow 3`` with index 5
* ``3 \rightarrow 2`` with index 3

So our second cycle is equal to $\{2,3,5\}$.

The third an final cycle is equal to the only remaining part, i.e. $\{6 \}$

We have now determined all cycles. The children aren obtained as follows:
* ``O_1[1,4,7,8] =  P_1[1,4,7,8] ``
* ``O_1[2,3,5] = P_2[2,3,5] ``
* ``O_1[6]  = P_1[6] ``

This leads to $O_1 = [1,5,2,4,3,6,7,8]$. In a similar fashion, but inverting the role of $P_1$ and $P_2$ one finds $O_2 = [8,2,3,1,5,6,4,7]$.

A downside of this type of crossover is that it can sometimes lead to the offspring being identical to the parents. You can verify this for $[3,4,8,2,7,1,6,5]$ & $[4,2,5,1,6,8,3,7]$.

##### Your own method
Come up with something that seems adequate...
"""

# ╔═╡ 1f5d255c-8576-11eb-1463-89f94258168d
begin
	"""
		mate(p1,p2;method=partialcrossover, kwargs...) 

	Function that handles the crossover for two different tours. The method keyword argument can be any of the functions you define as long as 1. the function exists 2. the function works in a similar way as the other ones (i.e. return two children from two parents.)

	Additional options for your own crossover implementation can be passed via the kwargs. You should document the options for each algorithm sperately.

	"""
	function mate(p1::Array{Int64,1},p2::Array{Int64,1}; method::Function=partialcrossover, kwargs...) 
			return method(p1,p2; kwargs...)
	end

	"""
		partialcrossover(p1, p2; kwargs...)

	Implementation of the partial crossover function for TSP routing problems.
	"""
	function partialcrossover(p1::Array{Int64,1},p2::Array{Int64,1}; kwargs...)
		"""
			deconflicter!(O::Array{Int64,1}, mapping::Dict)

		Function that deals with the conflicts after the initial assignments.
		"""
		function deconflicter(O::Array{Int64,1},p::Array{Int64,1}, mapping::Dict)
			for i in 1:length(O)
				if O[i] == 0
					cand = p[i]
					while cand in O
						cand = mapping[cand]
					end
					O[i] = cand
				end
			end
			return O
		end

		# choose cut points
		c1, c2 = sort(sample(1:length(p1),2, replace=false))
		# initiate offspring
		O1, O2 = [zeros(Int64,length(p1)) for _ in 1:2]
		# determine mapping 
		map12 = Dict(p1[i]=> p2[i] for i in c1:c2)
		map21 = Dict(value => key for (key, value) in map12)
		# set offspring initial values
		O1[c1:c2] = p2[c1:c2]
		O2[c1:c2] = p1[c1:c2]
		# fill non-conflicting positions
		for i in setdiff(1:length(p1),c1:c2)
			O1[i] = p1[i] in O1 ? 0 : p1[i]
			O2[i] = p2[i] in O2 ? 0 : p2[i]
		end
		# fill conflicting positions using the mapping
		O1, O2 = deconflicter(O1, p1, map21), deconflicter(O2, p2, map12)

		return O1, O2
	end

	"""
		cyclecrossover(p1, p2)

	Implementation of the cycle crossover function for TSP routing problems.
	"""
	function cyclecrossover(p1::Array{Int64,1},p2::Array{Int64,1})
		# initiate children
		o1=zeros(Int64,length(p1))
		o2=zeros(Int64,length(p2))

		# Pt. 1 - find all cycles

		# make hashtable: value => index
		h1 = Dict{Int64, Int64}(p1[i]=>i for i in 1:length(p1))
		h2 = Dict{Int64, Int64}(p2[i]=>i for i in 1:length(p2))
		# make hashtable to track usage in a cycle: index => bool
		u1 = Dict{Int64, Bool}(i=>false for i in 1:length(p1))
		# loop over parent creating cycles on the go
		# a cycle is stored in a Set because this is hashable. This is (much) faster than an array for ∉ check
		# the values stores in the cycles are the indices to be used
		cycles = Array{Set{Int64},1}()
		for i in 1:length(p1)      
			if !u1[h1[p1[i]]] # if not used yet, start a new cycle
				cycle = Set{Int64}()
				next = h1[p1[i]] # add initial values
				while next ∉ cycle
					push!(cycle, next)  # update cycle
					u1[next] = true     # update tracking
					next = h1[p2[next]] # next location
				end
				push!(cycles, cycle)
			else
				continue
			end    
		end

		# Pt. 2 - make children from cycles
		for i in 1:length(cycles)
			inds = collect(cycles[i])
			if isodd(i)
				o1[inds] = p1[inds]
				o2[inds] = p2[inds]
			else
				o1[inds] = p2[inds]
				o2[inds] = p1[inds]
			end
		end
		return o1, o2
	end


	"""
		mycrossover(p1, p2; myoption::Float64=0.5, kwargs...)

	Your own method that takes predefined options and also accepts others.
	"""
	function mycrossover(p1::Array{Int64,1},p2::Array{Int64,1}; myoption::Float64=0.5, kwargs...)
		error("Function <mycrossover> not defined yet... Get creative!")
	end

	nothing
end

# ╔═╡ f064b884-6bf2-4669-95d8-f4dfd8d297b2
begin
	# demo
	p1 = ['a';'b';'c']
	p2 = ['d';'e';'f']
	# pure mutation, no mutation, some mutation
	mate(p1,p2, pmut=1),mate(p1,p2, pmut=0), mate(p1,p2, pmut=0.1)
end

# ╔═╡ 16cbf73d-b556-472a-94bd-5c1b1888251b
"""
	textga(;goal::Array{Char,1}=goal, popsize::Int64=pop_size, 
                	 ngen::Int64=500, next::Float64=0.4, pmut=0.1, kwargs...)

From a population composed of random gene combinations, evolve towards a goal state. Returns
the current generation, the best individual and the ratio of the fitness of the best individual
with respect to the fitness of the goal state.

keyword arguments:
- goal: goal state
- popsize: number of individuals in the population
- ngen: maximum number of generations to consider
- next: percentage of individuals that is used for the next generation
- pmut: probability of mutation
"""
function textga(;goal::Array{Char,1}=goal, popsize::Int64=pop_size, 
                	 ngen::Int64=500, next::Float64=0.4, pmut=0.1, fitness::Function, kwargs...)
		# initiate population
		population = [rand(genes, length(goal)) for _ in 1:popsize]
		gen = 0
		for _ in 1:ngen
			# sort by fitness
			sort!(population; by=x->fitness(x), rev=true)
			if population[1] == goal
				break
			end
			gen += 1
			# select adequate parents
			goodparents = population[1: round(Int,popsize*next)]
			# make children
			new_generation = typeof(population)()
			for _ in 1:popsize
				push!(new_generation, mate(rand(goodparents), rand(goodparents),
											pmut=pmut ))
			end
			population = new_generation
		end


		return (gen, prod(population[1]), fitness(population[1])/fitness(goal))
end; nothing

# ╔═╡ 1499fad4-45f3-442e-b705-b83440f21ba5
textga(ngen=500, popsize=100, pmut=0.1, fitness=fitness_message)

# ╔═╡ 08ca0837-47f2-4b8c-9f88-9e52e86dfb3f
let
	n = 1000
	duration = Vector{Int}(undef,n)
	for i = 1:n
		duration[i],_,_ = textga(ngen=500, popsize=100, pmut=0.1, fitness=fitness_message)
	end
	histogram(duration, normalize=:pdf, xlabel="number of generations", ylabel="relative frequency", label="")
	
end

# ╔═╡ 7c794b5c-ecf0-4132-82af-4ece165cc158


# ╔═╡ 7c4ef658-8576-11eb-3f6d-d3587a1af7e8
begin
	let
		# small demo matching the explanation of the algorithm above. 
		# Matches the detailed example if you force c1, c2 = 4, 6 in the partialcrossover algorithm
		p1 = [3,4,8,2,7,1,6,5]
		p2 = [4,2,5,1,6,8,3,7]
		@btime mate($p1,$p2)
	end

	let
		# Longer routes (≈12x longer route and only ≈6x slower)
		N = 100
		p1 = collect(1:N)
		p2 = sample(p1,N,replace=false)
		@btime mate($p1,$p2);
	end

	let
		# demo from the method description
		p1 = [1,2,3,4,5,6,7,8]
		p2 = [8,5,2,1,3,6,4,7]
		@btime mate($p1,$p2, method=$cyclecrossover)
		@assert mate(p1,p2, method=cyclecrossover) ==  ([1,5,2,4,3,6,7,8],  [8,2,3,1,5,6,4,7])
	end

	let
		# demo for children equal to parents
		p1 = [3,4,8,2,7,1,6,5]
		p2 = [4,2,5,1,6,8,3,7]
		@btime mate($p1,$p2, method=$cyclecrossover)
		@assert mate(p1,p2, method=cyclecrossover) == (p1 ,p2)
	end

	let
		# Longer routes (≈12x longer route and only ≈7x slower)
		N = 100
		p1 = collect(1:N)
		p2 = sample(p1,N,replace=false)
		@btime mate($p1,$p2, method=$cyclecrossover)
	end
	nothing
end

# ╔═╡ b85012fb-3fde-48ba-bc93-eacb7c03dd65
md"""
### Local beam search
You start with k randomly gerated states. At each step you generate all successor states. You select the best k successors and repeat untill you find a solution.

We reconsider the problem used for simulated annealing, done with k = 4.
"""

# ╔═╡ 885afc3f-a5f9-4222-af16-b0043fea58cf
begin
	"""
		localbeamsearch(initstates, fitness::Function, successor::Function)

	Use local beam search starting from a set of initial states. The successors are generated by the `successor` function and 
	the fitness is evaluated by the `fitness` function.
	"""
	function localbeamsearch(initstates; fitness::Function, successor::Function, Nmax::Int=100)
		k = length(initstates)
		beststates = copy(initstates)
		for i = 1:Nmax
			sucessors = mapreduce(x->successor(x), vcat, beststates) # candidates
			beststates = sort!(sucessors, by=x->-fitness(x))[1:k] 	 # best k candidates
			if all(y -> y == first(beststates), beststates)			 # early stopping if all values are the same
				@warn "Stopped after $(i) iterations"
				break
			end
		end
		
		return beststates
	end
	nothing
end

# ╔═╡ 1a79eb5a-8577-11eb-168e-8fa9230f26fb
md"""
## Local search for CSPs
In the following we show how some of these methods can be use to solve a CSP. We start with the travelling salesman problem.
### TSP with simulated annealing
"""

# ╔═╡ c84f6af2-8568-11eb-06c4-53f7664294ac
begin
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
	function plotgraph(g::SimpleWeightedGraph, d::Dict;wf::Real=5,draw_MST::Bool=true)
		p = plot(legend=:bottomright)
		for e in edges(g)
			plot!([d[e.src][1], d[e.dst][1]],[d[e.src][2], d[e.dst][2]], 
				markershape=:circle, linewidth=e.weight*wf, color=:black, markeralpha = 0.5,
				linealpha=0.2, label="", markercolor=:black, markersize=5)
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

	Function that plots the final TSP solution
	"""
	function plotsol(fp::Array{Int64,1}, nodedict::Dict)
		p = plot()
		for i in 1:length(fp)-1
			x = [nodedict[fp[i]][1]; nodedict[fp[i+1]][1]]
			y = [nodedict[fp[i]][2]; nodedict[fp[i+1]][2]]
			plot!(x,y,color=:blue,label="", marker=:circle)
		end
		# close loop
		x = [nodedict[fp[1]][1]; nodedict[fp[end]][1]]
		y = [nodedict[fp[1]][2]; nodedict[fp[end]][2]]
		plot!(x,y,color=:red,label="", marker=:circle)
		xlims!(0,1)
		ylims!(0,1)
		return p
	end
	
	nothing
end

# ╔═╡ 46e468e0-8569-11eb-0a30-d11e814ccf7e
begin
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
	
	nothing
end

# ╔═╡ 7c5282fa-8569-11eb-0dcf-abc7b3b795ef
begin
	N = problemgenerator(15) 							# nodes
	nodedict = Dict( i => N[i] for i in 1:length(N)) 	# nodes dict
	A = distancematrix(N,2) 							# distance matrix using manhattan norm
	G = SimpleWeightedGraph(A) 							# graph
	p = plotgraph(G,nodedict;draw_MST=false) 			# layout of the current graph
end

# ╔═╡ eb27eada-8569-11eb-0739-7717087a5240
begin
	"""
		finit(A::Array{Float64,2})
	
	Generate an initial distribution based on a random order
	"""
	function finit(A::Array{Float64,2})
		return shuffle(1:size(A,1))
	end
	
	finit(A)
end

# ╔═╡ 1692c654-856a-11eb-1b02-77b0ab027c29
begin
	"""
		fitness(T::Array{Int64,1},A::Array{Float64,2})
	
	Given a tour, compute the negatice value of the tour distance based on the distance matrix
	"""
	function fitness(T::Array{Int64,1},A::Array{Float64,2})
		return -sum([A[T[i],T[i+1]] for i in 1:length(T)-1]) - A[T[1],T[end]] 
	end
	
	fitness(finit(A),A)
end

# ╔═╡ 3c58cc1a-856a-11eb-17bd-a31c244f6229
begin
	"""
		successors(T::Array{Int64,1}; nsuc::Int64=5)
	
	Determine successors for a given tour. The successors are based on a random permutation of partial tour, i.e. two locations in the tour are sampled. Then, the subtour delimited by those two locations is shuffled. In total this process is repeated `nsuc` times to generate multiple successors.
	"""
	function successors(T::Array{Int64,1}; nsuc::Int64=5)
		res = Array{Array{Int64,1},1}()
		for _ in 1:nsuc
			i,j = sample(1:length(T),2,replace=false,ordered=true)
			cand = copy(T)
			cand[i:j] = shuffle(T[i:j])
			push!(res, cand)
		end

		return res
	end
	nothing
end

# ╔═╡ 6c47688d-08b6-4878-98e4-24f535544b35
begin
	k = 4 					# number of beams
	xinit = rand(k) .*5 	# initial states
	
	xopt = localbeamsearch(xinit, fitness=x->fitness(x,f), successor=successors)
	plot(range(0,5, length=100),f.(range(0,5, length=100)),label="")
	scatter!(xinit,f.(xinit),label="start states",marker=(:circle, :green), markeralpha=0.5)
	scatter!(xopt,f.(xopt),label="optimal values",marker=(:circle, :red),size=(500,300), legend=:bottomright, markeralpha=0.3)
	title!("Local beam search results")
end

# ╔═╡ d6dab40b-d49e-4e1d-8504-ad5b5a6cc762
successors(finit(A))

# ╔═╡ 7ab96168-856c-11eb-166a-b7531078ac90
begin
	"""
		simulated_annealing(args; kwargs...)
	
	Simulated annealing implementation for TSP

	Since this is stochastic (sort of a random walk), it will not always reach the optimal value, therefore, one typically runs several iterations and the best one is retained.

	The probabiliy of accepting a value that lead to a decrease in energy is given by exp(ΔE/T). In the literature you might also find  1/(1 + exp(ΔE/T)) ∈ [1/2, 1] as a treshold.
	"""
	function simulated_annealing(A::Array{Float64,2}, init::Array{Int64,1}; 
								tmax=1000, T0::Float64=273.0, α::Float64=0.005,
								fitness::Function=fitness,nsuc::Int64=10, kwargs...)
		current = init
		next = init
		T = T0
		best = init
		for t = 1:tmax
			T = scheduletemp(t,T0=T0)
			if T == 0
				return current
			else
				next = rand(successors(current, nsuc=nsuc))
				ΔE = fitness(next, A) - fitness(current, A)
				if ΔE > 0 
					current = next
				else
					if rand() < exp(ΔE/T)
						current = next
					end
				end
				if fitness(current, A) > fitness(best, A)
					best = current
				end
			end
		end
		return best
	end
	
	"""
	wrapper function for multiple SA iterations

	Returns the best n found value for N runs of simulated annealing.
	"""
	function SAsearch(A::Array{Float64,2}, n::Int64=1, N::Int64=10;kwargs...)
		sort!([simulated_annealing(A, finit(A); kwargs...) for _ in 1:N], 
				by=x->kwargs[:fitness](x,A),rev=false)[1:n]
	end

	nothing
end

# ╔═╡ 728b94da-8566-11eb-1b02-e94c64424845
begin
	# general options
	options = Dict(	:tmax=>1000, :T0=>273.0, :α=>0.005, 
               		:fitness=>fitness, :optimfun=>f)
	x0 = 0.0;
	# quick illustration
	simulated_annealing(x0; options...)
end

# ╔═╡ 0ce6432c-8567-11eb-1465-432d66051360
begin
	n = 20;
	res = SAsearch(x0,n,n;options...)
	plot(range(0,5, length=100),f.(range(0,5, length=100)),label="")
	scatter!([x0],[f.(x0)],label="start",marker=(:cross, :green))
	scatter!([res],[f.(res)],label="optimal values",marker=(:circle, :red),size=(500,300), legend=:bottomright, markeralpha=0.3)
	title!("Simulated annealing results")
end

# ╔═╡ 95a66570-33c2-44c2-bf2d-61b7e9f44b3f
simulated_annealing(A, finit(A)) # single solution

# ╔═╡ b1cdd85d-98f5-4ee9-b434-10ca86b063ff
SAsearch(A, 10, 10, fitness=fitness) # multiple solutions

# ╔═╡ ef64ca86-856c-11eb-0cd0-178565e85576
let
	options = Dict(:tmax=>1000, :T0=>273.0, :α=>0.005)
	N_tot = 1000
	tspres = SAsearch(A, 10, N_tot; fitness=fitness, options...)[1]
	init = finit(A)
	println("Cost of a random trip: $(round(-fitness(init,A), digits=2))")
	println("Cost of a optimised trip: $(round(-fitness(tspres,A), digits=2))")
	plotsol(tspres, nodedict)
	title!("TSP solution using simulated annealing")
end

# ╔═╡ 57726a64-856d-11eb-2b37-877202c47ede
let
	# Larger example
	n = 50 # number of nodes
	N = problemgenerator(n) # nodes
	options = Dict(:tmax=>1000, :T0=>273.0, :α=>0.005, :fitness=>fitness)
	nodedict = Dict( i => N[i] for i in 1:length(N)) # nodes dict
	A = distancematrix(N,2) # distance matrix using manhattan norm
	init = finit(A)
	
	res = SAsearch(A,1,500; options...)[1]
	println("Cost of a random trip: $(-fitness(init,A))")
	println("Cost of a optimised trip: $(-fitness(res,A))")
	plotsol(res, nodedict)
end

# ╔═╡ d70478c8-856d-11eb-3f27-0b565955b6ed
let
	# trajectory cost distribution
	res = Array{Float64,1}()
	for _ in 1:50
		push!(res, -fitness(SAsearch(A;options...)[1], A))
	end
	histogram(res, normalize=:pdf, alpha=0.5,label="PDF optimised tour cost")
	scatter!([-fitness(finit(A),A)],[0],label="initial non-optimised cost")
	xlabel!("Tour cost")
	ylabel!("PDF")
	title!("Histogram for tour cost ($(size(A,1)) cities)")
end

# ╔═╡ 95d68906-eb8d-4bf9-aa0a-1ff1424555bc
md"""
### TSP with genetic algorithm
Let's try to solve the TSP problem again, but this time, using a genetic algorithm.

We should think about:
* representation
* an initial configuration - e.g. a tour by using a nearest neigbor approach or purely random
* a fitness function 
* a selection method
* cross-over method  ``\rightarrow`` what is the difference with the text application?
* mutation rate and method ``\rightarrow`` is this required? why (not)?
* when do you stop making new generations?
* ``\dots``


Tasks:
* Come up with your own suggestions for cross-over/mutation
* Implement the genetic algorithm to solve the TSP
* Compare the performance of different cross-over algorithms. Given that there is some randomness involved, a statistical approached might be more appropriate. Since the initial population will most likely be chosen at random, you should impose the random seed to make sure you have the same starting point for the comparisons.
* Compare the scalability with what you know from previous practical sessions (tree search etc.) What is better and what is worse?

"""

# ╔═╡ 1182ad00-e2cf-4899-a9f4-765049f8491d
"""
	TSP_ga(d::Array{Float64,2}; 
					popsize::Int64=30, ngen::Int64=500, next::Float64=0.4,
					method::Function=partialcrossover,
					fitness::Function=fitness, kwargs...)

Genetic algorithm implementation to solve a TSP. Takes a distance matrix as input and starts from a population of randomly generated tours.

keyword arguments:
- popsize: number of individuals in the population
- ngen: total number of generations
- next: percentage of individuals that is used for the next generation
- method: function used for the crossover
- fitness: function used for the fitness
"""
function TSP_ga(d::Array{Float64,2}; 
					popsize::Int64=30, ngen::Int64=500, next::Float64=0.4,
					method::Function=partialcrossover, 
					fitness::Function=fitness, kwargs...)
		# initiate population
		n = size(d,1)
		population = [sample(1:n,n,replace=false) for _ in 1:popsize]
		gen = 0
		# go over the generations
		for _ in 1:ngen
			# sort by fitness
			sort!(population; by=x->fitness(x,d), rev=true)
			#@show "current tour cost: $(fitness(population[1],d))"
			# include a stopcondition?
			gen += 1
			# select adequate parents
			goodparents = population[1: round(Int,popsize*next)]
			# make children
			new_generation = typeof(population)()
			for _ in 1:2:popsize
				p1,p2 = rand(goodparents,2)
				offspring = mate(p1,p2;method=method, kwargs...) 
				append!(new_generation, offspring)
			end
			population = new_generation
		end

		return (gen, population[1], fitness(population[1],d))
end;nothing

# ╔═╡ fa780328-8576-11eb-3a16-b7109ffbad63
let
	n = 50 # number of nodes
	N = problemgenerator(n) # nodes
	nodedict = Dict( i => N[i] for i in 1:length(N)) # nodes dict
	A = distancematrix(N,1) # distance matrix using manhattan norm
	G = SimpleWeightedGraph(A) # graph
	begin
    	D = Dict()
    	methods = [cyclecrossover, partialcrossover]
		for method in methods
			res = []
			for _ in 1:1000
				(_, tour, cost) = TSP_ga(A, ngen=20, popsize=n,method=method)
				push!(res,(tour,cost))
			end
			D[method] = [x[2] for x in res]
		end
	end
	
	# illustration
	histogram()
	for (key, val) in D
		histogram!(-val, label=string(key), normalize=:pdf, alpha=0.5)
	end
	scatter!([-fitness(finit(A),A)],[0],label="initial non-optimised cost")
	xlabel!("Tour cost")
	ylabel!("PDF")
	title!("Histogram for tour cost ($(size(A,1)) cities)")
end

# ╔═╡ 6830a340-bd19-4aa1-a30e-35248bfd4550
md"""
### TSP with local beam search
We can define a neighbor as a pairwise switch between two links in the tour. By doing so, we could in principle even (but this is not guaranteed) find  the best solution.
"""

# ╔═╡ 8ce5ab61-e7ce-42a7-95e6-dafd165ab38b
begin
	"""
		TSPsuccessors(state, n::Int=2)

	Generate successors for TSP problem. Picks two edges at random and switches them. Generates `n` successors.
	"""
	function TSPsuccessors(state::T, n::Int=2) where T
		ns = length(state)
		next = Vector{T}(undef, n)
		for k = 1:n
			i,j = 0, ns
			#i = 3; j=6
			while (j-i <= 1) || (j >= ns)# (j-i <= 1)# && (j >= length(state) - 1)
				i,j = sample(1:ns,2,replace=false,ordered=true) # startlocations	
			end
			next[k] = vcat( state[1:i], 
							state[j], 
							state[i+2:j-1], 
							state[i+1], 
							state[(j+1 > ns ? end : j+1) : end])
		end
		
		return next
	end
	nothing
end

# ╔═╡ dfda3296-0c6a-43c7-94b6-814989ed1c4f
TSPsuccessors([1; 2; 5; 3; 4; 6; 7; 8; 9])

# ╔═╡ b11e49d0-508b-45e3-951a-7f0d49e94e52
begin
	TSPprob = problemgenerator(20) 										# node coordinates
	nodedict_TSP = Dict(i => TSPprob[i] for i in 1:length(TSPprob)) 	# nodes dict
	A_TSP = distancematrix(TSPprob,2) 									# distance matrix using manhattan norm
	G_TSP = SimpleWeightedGraph(A_TSP) 									# graph
	kTSP = 6
	TSPinitstates = [shuffle!(collect(1:length(TSPprob))) for _ = 1:kTSP]  # initial states
	
	TSP_sol = localbeamsearch(TSPinitstates; 	fitness=x->fitness(x,A_TSP), 
												successor=x->TSPsuccessors(x, kTSP), Nmax=3000)
end

# ╔═╡ 58045435-345e-4bf9-9865-e0c39b52f0a1
begin
	plotsol(TSP_sol[1], nodedict_TSP)
	title!("TSP with local beam search\ninitial cost: $(round(-fitness(TSPinitstates[1],A_TSP), digits=2))\nfinal cost: $(round(-fitness(TSP_sol[1],A_TSP), digits=2))")
end

# ╔═╡ 00461d06-409a-4699-8cc3-4d231f0c5b26
md"""
We can also look at the distribution over multiple solutions:
"""

# ╔═╡ 6fb224a6-ee9f-4480-a69e-a4be406c620b
let
	TSPprob = problemgenerator(20) 										# node coordinates
	nodedict_TSP = Dict(i => TSPprob[i] for i in 1:length(TSPprob)) 	# nodes dict
	A_TSP = distancematrix(TSPprob,2) 									# distance matrix using manhattan norm
	G_TSP = SimpleWeightedGraph(A_TSP) 									# graph
	kTSP = 6
	TSPinitstates = [shuffle!(collect(1:length(TSPprob))) for _ = 1:kTSP]  # initial states

	totalruns = 200
	res = Vector{Float64}(undef,  totalruns)
	for k = 1:totalruns
		TSP_sol = localbeamsearch(TSPinitstates; 	fitness=x->fitness(x,A_TSP), 
												successor=x->TSPsuccessors(x, kTSP), Nmax=3000)
		res[k] = -fitness(TSP_sol[1], A_TSP)
	end

	# illustration
	histogram()
	histogram(res, label="distribution of optimised costs")
	scatter!(map(x-> -fitness(x,A_TSP), TSPinitstates),[1],label="initial non-optimised costs")
	xlabel!("Tour cost")
	ylabel!("counts")
	title!("Histogram for tour cost ($(size(A_TSP,1)) cities)")	
end

# ╔═╡ 4b3ead16-31a0-4c0f-b0f0-e0537c34dc3f
md"""
## Additional applications
Use one of the local search methods to solve the following problems
* N-queens
* Graph coloring
For each problem, you should think about how to define fitness and how to define a neighbor state.
"""

# ╔═╡ 499523bd-716f-4141-8b89-fb9d4cc05598
begin
	# demo for graphs
	GG = Graphs.Graph(6)
	add_edge!(GG, 1,2)
	add_edge!(GG, 1,5)
	add_edge!(GG, 2,3)
	add_edge!(GG, 2,5)
	add_edge!(GG, 3,5)
	add_edge!(GG, 3,4)
	add_edge!(GG, 4,5)

	const graphcolors = [:red, :black, :blue, :white]

	solinitgraphc = rand(graphcolors, 6)

	function graphconflicts(state; graph::SimpleGraph)
		nc = 0
		for e in edges(graph)
			if state[e.src] == state[e.dst]
				nc += 1
			end
		end
		return nc
	end

	function graphsuccessorstates(state; graph::SimpleGraph)
		# total number of successor state: N_nodes* (N_colors - 1)
		successors = Vector{Vector{Symbol}}()
		for i in vertices(graph)
			for color in graphcolors
				if color ≠ state[i]
					next = copy(state)
					next[i] = color
					push!(successors, next)
				end
			end
		end
		return successors
	end

	gcres = localbeamsearch([rand(graphcolors, 6) for _ in 1:4], 
							fitness=x -> -graphconflicts(x; graph=GG), 
							successor=x->graphsuccessorstates(x; graph=GG))
end

# ╔═╡ 6878c69d-5fe8-4e6b-9246-270b11e15382


# ╔═╡ Cell order:
# ╟─8a4c3b83-4b66-41d6-a075-be34c74b456f
# ╟─bca1c92c-8561-11eb-0420-6f0123bacb7f
# ╟─774fa124-8560-11eb-2317-53b0d106b67a
# ╟─eb014126-bd42-4581-9cf8-324dc093f964
# ╟─87a06bd6-8f60-4bc8-8eee-d52f1636b56f
# ╠═9db116bd-7ddc-42ef-ba26-b9538a9b89c1
# ╟─d47e035d-c756-4510-aa10-328cbdf2f8a6
# ╠═d5019d8a-8561-11eb-2a43-a9ce616532f5
# ╠═728b94da-8566-11eb-1b02-e94c64424845
# ╟─0ce6432c-8567-11eb-1465-432d66051360
# ╟─92d33c5f-546b-4667-803d-773844a9fe7e
# ╠═e9cc9a3b-4353-45da-991b-2045d7aef756
# ╠═777949bf-590e-4f03-93e4-099f28273bbd
# ╠═6ec4f09d-6179-4636-9e16-a7a6aff9fcef
# ╠═f064b884-6bf2-4669-95d8-f4dfd8d297b2
# ╠═16cbf73d-b556-472a-94bd-5c1b1888251b
# ╠═1499fad4-45f3-442e-b705-b83440f21ba5
# ╟─9eabd1b1-e162-4194-bc96-c2d419a76b79
# ╟─08ca0837-47f2-4b8c-9f88-9e52e86dfb3f
# ╟─075bebb8-73a8-4a89-a088-3a23535be67d
# ╠═1f5d255c-8576-11eb-1463-89f94258168d
# ╠═7c794b5c-ecf0-4132-82af-4ece165cc158
# ╠═7c4ef658-8576-11eb-3f6d-d3587a1af7e8
# ╟─b85012fb-3fde-48ba-bc93-eacb7c03dd65
# ╠═885afc3f-a5f9-4222-af16-b0043fea58cf
# ╠═6c47688d-08b6-4878-98e4-24f535544b35
# ╟─1a79eb5a-8577-11eb-168e-8fa9230f26fb
# ╟─c84f6af2-8568-11eb-06c4-53f7664294ac
# ╠═46e468e0-8569-11eb-0a30-d11e814ccf7e
# ╠═7c5282fa-8569-11eb-0dcf-abc7b3b795ef
# ╠═eb27eada-8569-11eb-0739-7717087a5240
# ╠═1692c654-856a-11eb-1b02-77b0ab027c29
# ╠═3c58cc1a-856a-11eb-17bd-a31c244f6229
# ╠═d6dab40b-d49e-4e1d-8504-ad5b5a6cc762
# ╠═7ab96168-856c-11eb-166a-b7531078ac90
# ╠═95a66570-33c2-44c2-bf2d-61b7e9f44b3f
# ╠═b1cdd85d-98f5-4ee9-b434-10ca86b063ff
# ╠═ef64ca86-856c-11eb-0cd0-178565e85576
# ╠═57726a64-856d-11eb-2b37-877202c47ede
# ╟─d70478c8-856d-11eb-3f27-0b565955b6ed
# ╟─95d68906-eb8d-4bf9-aa0a-1ff1424555bc
# ╠═1182ad00-e2cf-4899-a9f4-765049f8491d
# ╟─fa780328-8576-11eb-3a16-b7109ffbad63
# ╟─6830a340-bd19-4aa1-a30e-35248bfd4550
# ╠═8ce5ab61-e7ce-42a7-95e6-dafd165ab38b
# ╠═dfda3296-0c6a-43c7-94b6-814989ed1c4f
# ╠═b11e49d0-508b-45e3-951a-7f0d49e94e52
# ╠═58045435-345e-4bf9-9865-e0c39b52f0a1
# ╟─00461d06-409a-4699-8cc3-4d231f0c5b26
# ╠═6fb224a6-ee9f-4480-a69e-a4be406c620b
# ╟─4b3ead16-31a0-4c0f-b0f0-e0537c34dc3f
# ╠═499523bd-716f-4141-8b89-fb9d4cc05598
# ╠═6878c69d-5fe8-4e6b-9246-270b11e15382
