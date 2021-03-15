### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ bca1c92c-8561-11eb-0420-6f0123bacb7f
begin
	# dependencies
	using Plots
	using LightGraphs, SimpleWeightedGraphs, LinearAlgebra, Random
	import StatsBase:sample
end

# ╔═╡ 774fa124-8560-11eb-2317-53b0d106b67a
md"""
# Local Search
"""

# ╔═╡ 2b5e6876-8561-11eb-0685-dfe1cca39c45
md"""
## Simulated annealing
Simulated annealing is similar to hill climbing, but it allows downhill moves. There are several aspects to consider. 
* What is considered a neighbor? And how to chose them?
* How to evaluate the energy (i.e. the fitness)?
* How to let the temperature evolve. Several options exist here:
    - exponential decrease: $T = T_0(1-\alpha)^k$, where $\alpha$ is the ‘cooling rate’ and $k$ is the iteration
    - fast decrease: $T = T_0/k$
    - boltzmann decrease: $T = T_0/log(k)$   
"""

# ╔═╡ 4e190a2e-8561-11eb-3759-795e3bb4d8d2
md"""
To illustrate the principle, let's try to find the maximum of 
\\[ f(x) = -\dfrac{1}{4}x^4 +  \dfrac{7}{3} x^3 - 7x^2 + 8x \\]
in the interval $[0, 5]$. You can observe that there is a global maximum in $x=4$, however, there is a local maximum in $x=1$.
"""

# ╔═╡ b49f0528-8561-11eb-36a7-fd42574edc81
begin
	f = x-> -1/4*x^4 + 7/3 *x^3 - 7*x^2 + 8*x
	x = range(0,5, length=100);
	plot(x,f.(x),label="",size=(300,200))
end

# ╔═╡ d5019d8a-8561-11eb-2a43-a9ce616532f5
begin
	"""
	Evaluate the fitness of a specific point
	"""
	function fitness(x::Float64,f::Function)
		return f(x)
	end

	"""
	Determine successors for a given location (with a fixed step size)
	You could include automatic step size modification if you want.
	"""
	function successors(x::Float64; step=0.1)
		return x .+ step*[-1; 1]
	end

	"""
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
	
	The probabiliy of accepting a value that lead to a decrease in energy is given by exp(ΔE/T), which is between 0 (temp = 0) and one (ΔE for any T ≠ 0).

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
	function simulated_annealing(x0; 
				tmax=1000, T0::Float64=273.0, α::Float64=0.005,
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
	
	Wrapper function for multiple SA iterations. Returns the best n found value for N runs of simulated annealing.
	"""
	function SAsearch(x0, n::Int64=1, N::Int64=10;kwargs...)
		return sort!([simulated_annealing(x0; kwargs...) for _ in 1:N], 
					 by=x->kwargs[:fitness](x,kwargs[:optimfun]),
					 rev=true)[1:n]
	end
end

# ╔═╡ 9c2e2032-8568-11eb-103f-fdc57c11ab1f
md"""
### Solving the TSP problem with simulated annealing
Use simulated annealing to solve the travelling saleman problem again. You will need to think in particalar about:
* an appropriate inital solution (e.g. a tour by using a nearest neigbor approach or purely random)
* generating appropriate neigbors (e.g. by flipping a subpart of a tour)
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
	
	"plot functions"
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
end

# ╔═╡ 7c5282fa-8569-11eb-0dcf-abc7b3b795ef
begin
	N = problemgenerator(15) # nodes
	nodedict = Dict( i => N[i] for i in 1:length(N)) # nodes dict
	A = distancematrix(N,2) # distance matrix using manhattan norm
	G = SimpleWeightedGraph(A) # graph
	p = plotgraph(G,nodedict;draw_MST=false) # layout of the current graph
end

# ╔═╡ eb27eada-8569-11eb-0739-7717087a5240
begin
	"""
		finit(A::Array{Float64,2})
	
	Generate an initial distribution based and a random order
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
	
	Given a tour, compute the tour distance based on the distance matrix
	"""
	function fitness(T::Array{Int64,1},A::Array{Float64,2})
		return -sum([A[T[i],T[i+1]] for i in 1:length(T)-1]) - A[T[1],T[end]] 
	end
	
	fitness(finit(A),A)
end

# ╔═╡ 3c58cc1a-856a-11eb-17bd-a31c244f6229
begin
	"""
	Determine successors for a given tour (simple random permutation of partial tour)
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
	successors(finit(A))
end

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
	

	simulated_annealing(A, finit(A))
end

# ╔═╡ 934aa9e4-856e-11eb-1924-fd1df8fd3fe2
md"""
## Genetic algorithms
a genetic algorithm (GA) is inspired by the process of natural selection. In order to implement this you need the following items:
* an initial population (e.g. random)
* a fitness function that allows you to rank the population members
* a selection method (e.g. roulette based proportions, ‘elite selection’)
* a cross-over method (cutting; 50/50 per gene etc.)
* mutation rate and method
* \\(\dots\\)
"""

# ╔═╡ 8753fd9a-856f-11eb-2d91-13657e302031
md"""
### Toy example
Let's try write a message by using a genetic algorithm. E.g "172 Pol evolved!" starting from a random set of characters (a-zA-Z0-9. !)
"""

# ╔═╡ 48c05e06-8570-11eb-0367-1f3db0aa20d0
begin
	chars = vcat(65:90,97:122, 48:57,[Int(' ');Int('.'); Int('!')])
	const genes = [Char(i) for i = chars]
	const pop_size = 100
	const goal = collect("172 Pol evolved!");
end

# ╔═╡ 6faf987e-8570-11eb-1861-7de0f661bd63
begin
	function fitness(p::Array{Char,1}; goal=goal)
		sum(p.==goal)
	end

	testcase = rand(genes,length(goal))
	testcase, fitness(testcase)
end

# ╔═╡ 728b94da-8566-11eb-1b02-e94c64424845
begin
	options = Dict(	:tmax=>1000, :T0=>273.0, :α=>0.005, 
               		:fitness=>fitness, :optimfun=>f)
	x0 = 0.0;
	simulated_annealing(x0; options...)
end

# ╔═╡ a72d7c3c-8569-11eb-36a3-cb68ab2622cf
let
	n = 10;
	res = SAsearch(x0,n;options...)
end

# ╔═╡ 0ce6432c-8567-11eb-1465-432d66051360
begin
	n = 10;
	res = SAsearch(x0,n;options...)
	plot(x,f.(x),label="")
	scatter!([x0],[f.(x0)],label="start",marker=(:circle, :green),size=(300,200), legend=:bottom)
	scatter!([res],[f.(res)],label="optimal values",marker=(:circle, :red),size=(300,200), legend=:bottom)
end

# ╔═╡ ef64ca86-856c-11eb-0cd0-178565e85576
let
	options = Dict(:tmax=>1000, :T0=>273.0, :α=>0.005, :fitness=>fitness)
	@show res = SAsearch(A; options...)[1];
	init = finit(A)
	println("Cost of a random trip: $(-fitness(init,A))")
	println("Cost of a optimised trip: $(-fitness(res,A))")
	plotsol(res, nodedict)
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
	
	@time res = SAsearch(A; options...)[1]
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

# ╔═╡ 9cc2fa40-8570-11eb-2eba-c15433dcb90e
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
	
	# demo
	p1 = ['a';'b';'c']
	p2 = ['d';'e';'f']
	mate(p1,p2)
end

# ╔═╡ 47153f8e-8572-11eb-1310-c932bd478721
md"""
### Some ideas for crossover:
Below you can find some algorithms that have proven to work quite well for this type of application. Both of these are described and an example and an implementation is provided. You are also invited to come up with a method yourself if you think you have a good idea.

#### The partially mapped crossover 
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


#### The cycle crossover 
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

#### Your own method
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
end

# ╔═╡ fd95d464-8570-11eb-1a38-879d8079d127
begin
	function textga(;goal::Array{Char,1}=goal, popsize::Int64=pop_size, 
                	 ngen::Int64=500, next::Float64=0.4, pmut=0.1, kwargs...)
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
		@info "stopped after $(gen) generations"

		return (gen, prod(population[1]), fitness(population[1])/fitness(goal))
	end

	textga(ngen=500, popsize=100, pmut=0.1)
end

# ╔═╡ 7c4ef658-8576-11eb-3f6d-d3587a1af7e8
begin
	let
		# small demo matching the explanation of the algorithm above. 
		# Matches the detailed example if you force c1, c2 = 4, 6 in the partialcrossover algorithm
		p1 = [3,4,8,2,7,1,6,5]
		p2 = [4,2,5,1,6,8,3,7]
		@time mate(p1,p2)
	end

	let
		# Longer route does not necessarily imply greater time for offspring generation
		# with the partial crossover algorithm (why could that be?)
		N = 100
		p1 = collect(1:N)
		p2 = sample(p1,N,replace=false)
		@time mate(p1,p2);
	end

	let
		# demo from the method description
		p1 = [1,2,3,4,5,6,7,8]
		p2 = [8,5,2,1,3,6,4,7]
		@time res = mate(p1,p2, method=cyclecrossover)
		@assert res ==  ([1,5,2,4,3,6,7,8],  [8,2,3,1,5,6,4,7])
	end

	let
		# demo for children equal to parents
		p1 = [3,4,8,2,7,1,6,5]
		p2 = [4,2,5,1,6,8,3,7]
		@time res = mate(p1,p2, method=cyclecrossover)
		@assert res == (p1 ,p2)
	end

	let
		# Longer routes scales nicely for cyclecrossover as well (≈10x longer route and only ≈2x slower)
		N = 100
		p1 = collect(1:N)
		p2 = sample(p1,N,replace=false)
		@time res = mate(p1,p2, method=cyclecrossover);
	end
end

# ╔═╡ 8b77cf1c-8571-11eb-2ea7-531abac106cc
md"""
### Genetic algorithm - TSP
Now that you are familiar with a genetic algorithm, let's try to solve the TSP problem again, but this time, using a genetic algorithm.

We should think about:
* representation
* an initial configuration
* a fitness function 
* a selection method (e.g. roulette based proportions, ‘elite selection’ etc.)
* cross-over method: (what is the difference with the text application?)
* mutation rate and method (is this required? why (not)?)
* when do you stop making new generations?
* \\( \dots \\)


Tasks:
* Come up with your own suggestions for cross-over/mutation
* Implement the genetic algorithm to solve the TSP
* Compare the performance of the tree cross-over algorithms to deal with eachother. Give that there is some randomness involved, a statistical approached might be more appropriate. Since the initial population will most likely be chosen at random, you should impose the random seed to make sure you have the same starting point for the comparisons.
* Compare the scalability with what you know from previous practical sessions (tree search etc.) What is better and what is worse?
"""

# ╔═╡ 712cb76c-8576-11eb-37c3-87beed1c3fa1
begin
	function TSP_ga(d::Array{Float64,2}; 
					popsize::Int64=30, ngen::Int64=500, next::Float64=0.4,
					method::Function=partialcrossover, kwargs...)
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
	end
end

# ╔═╡ fa780328-8576-11eb-3a16-b7109ffbad63
let
	n = 50 # number of nodes
	N = problemgenerator(n) # nodes
	nodedict = Dict( i => N[i] for i in 1:length(N)) # nodes dict
	A = distancematrix(N,1) # distance matrix using manhattan norm
	@info fitness(finit(A), A)
	G = SimpleWeightedGraph(A) # graph
	@time begin
    	D = Dict()
    	methods = [cyclecrossover, partialcrossover]
		for method in methods
			res = []
			for _ in 1:100
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

# ╔═╡ 1a79eb5a-8577-11eb-168e-8fa9230f26fb


# ╔═╡ Cell order:
# ╟─774fa124-8560-11eb-2317-53b0d106b67a
# ╠═bca1c92c-8561-11eb-0420-6f0123bacb7f
# ╟─2b5e6876-8561-11eb-0685-dfe1cca39c45
# ╟─4e190a2e-8561-11eb-3759-795e3bb4d8d2
# ╠═b49f0528-8561-11eb-36a7-fd42574edc81
# ╠═d5019d8a-8561-11eb-2a43-a9ce616532f5
# ╠═728b94da-8566-11eb-1b02-e94c64424845
# ╠═a72d7c3c-8569-11eb-36a3-cb68ab2622cf
# ╠═0ce6432c-8567-11eb-1465-432d66051360
# ╟─9c2e2032-8568-11eb-103f-fdc57c11ab1f
# ╟─c84f6af2-8568-11eb-06c4-53f7664294ac
# ╠═46e468e0-8569-11eb-0a30-d11e814ccf7e
# ╠═7c5282fa-8569-11eb-0dcf-abc7b3b795ef
# ╠═eb27eada-8569-11eb-0739-7717087a5240
# ╠═1692c654-856a-11eb-1b02-77b0ab027c29
# ╠═3c58cc1a-856a-11eb-17bd-a31c244f6229
# ╠═7ab96168-856c-11eb-166a-b7531078ac90
# ╠═ef64ca86-856c-11eb-0cd0-178565e85576
# ╠═57726a64-856d-11eb-2b37-877202c47ede
# ╠═d70478c8-856d-11eb-3f27-0b565955b6ed
# ╠═934aa9e4-856e-11eb-1924-fd1df8fd3fe2
# ╟─8753fd9a-856f-11eb-2d91-13657e302031
# ╠═48c05e06-8570-11eb-0367-1f3db0aa20d0
# ╠═6faf987e-8570-11eb-1861-7de0f661bd63
# ╠═9cc2fa40-8570-11eb-2eba-c15433dcb90e
# ╠═fd95d464-8570-11eb-1a38-879d8079d127
# ╟─47153f8e-8572-11eb-1310-c932bd478721
# ╠═1f5d255c-8576-11eb-1463-89f94258168d
# ╠═7c4ef658-8576-11eb-3f6d-d3587a1af7e8
# ╟─8b77cf1c-8571-11eb-2ea7-531abac106cc
# ╠═712cb76c-8576-11eb-37c3-87beed1c3fa1
# ╠═fa780328-8576-11eb-3a16-b7109ffbad63
# ╠═1a79eb5a-8577-11eb-168e-8fa9230f26fb
