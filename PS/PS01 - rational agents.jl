### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 88af7dc6-70fd-11eb-3af2-f7a194340290
begin
	using PlutoUI, Logging
	# set global loglevel
	Logging.global_logger(ConsoleLogger(stderr, Logging.LogLevel(-5000)))
	Logging.disable_logging(Logging.LogLevel(-1001))  
end

# ╔═╡ bb6ed424-70fc-11eb-2a2a-ff0556bb7cda
md"""
# Rational agents

## Our first agent
Consider the following example from the book: a simple vacuum-cleaner agent that cleans a square if it is dirty and moves to the other square if not. We consider the assume the following:
* Performance measure: we awards one point for each clean square at each time step, over a “lifetime” of 1000 time steps.
* Environment: the “geography” of the environment is known a priori but the dirt distribution and the initial location of the agent are not. Clean squares stay clean and sucking cleans the current square. The Left and Right actions move the agent left and right except when this would take the agent outside the environment, in which case the agent remains where it is.
* Actuators: the only available actions are Left, Right, and Suck.
* Sensors: the agent correctly perceives its location and whether that location contains dirt.

$(PlutoUI.LocalResource("./img/vacuum.png", :width => 500, :align => "middle"))

Questions:

1. How many world states are there?
2. Identify the goal states amongst those from q1.
3. Implement a performance-measuring environment simulator for the vacuum-cleaner world. Thinks about a modular implementation so that the sensors, actuators, and environment characteristics (size, shape, dirt placement, etc.) can be changed easily. 
4. What is the value of the performance measure for each world state? Confirm your simulation returns the expected value.
"""

# ╔═╡ 9fd46218-70fe-11eb-1ddd-132f13e0611c
begin
	# build the world
	struct world
		locations::Dict{Tuple{Int64,Int64}, Bool}
	end

	function world(locations::Array{Tuple{Int64,Int64}})
		return world(Dict((location => rand(Bool) for location in locations)))
	end

	function world(locations::Array{Tuple{Int64,Int64}}, dirt::Array{Bool,1})
		if length(locations) == length(dirt)
			return world(Dict(zip(locations, dirt)))
		end
	end

	mutable struct reflex_vacuum
		w::world
		percept::Tuple{Tuple{Int64, Int64}, Bool}
	end

	# include the actions
	function update!(v::reflex_vacuum)
		pos = v.percept[1]
		status = v.percept[2]

		@debug "current pos: $(pos), status: $(status)"
		if status == true # if dirty => clean
			@debug "status is dirty, cleaning..."
			v.w.locations[pos] = false
			v.percept = (pos, false)
		else # if clean => move
			pos = rules[pos]
			v.percept = (pos, v.w.locations[pos]) # update sensor
		end
	end

	function runprog(v::reflex_vacuum, n::Int=1000)
		score = 0
		for _ in 1:n
			score += sum(.!(values(v.w.locations)))
			update!(v)
		end
		return score
	end


	# __main__

	# set the rules
	const rules = Dict((0,0)=>(1,0), (1,0)=>(0,0))

	@info "Starting..."
	for layout in Base.Iterators.product([true, false],[true, false])
		for startpos in [(0,0),(1,0)]
			# initiate the world
			w = world([(0,0),(1,0)],[i for i in layout])
			# initiate reflex_vacuum
			v = reflex_vacuum(w, (startpos, w.locations[startpos]))
			# scoring
			score = runprog(v)
			# output msg
			@info "layout: $(layout), startpos: $(startpos), score: $(score)"
		end
	end
	@info "Finished"
end

# ╔═╡ ab564aec-70ff-11eb-0d36-cd95fb8def23
md"""
## Modified agent
Consider a modified version of the vacuum environment, in which the agent is penalized one point for each movement.

1. Can a simple reflex agent be perfectly rational for this environment? Explain.
2. What about a reflex agent with state?
3. How do your answers to a and b change if the agent’s percepts give it the clean/dirty status of every square in the environment?
"""

# ╔═╡ f2915910-70ff-11eb-2f60-e9963db3aaeb
md"""
## Another modified agent
Consider a modified version of the simple vacuum environment in which the geography of the environment — its extent, boundaries, and obstacles — is unknown, as is the initial dirt configuration. (The agent can go Up and Down as well as Left and Right.)

1. Can a simple reflex agent be perfectly rational for this environment? Explain.
2. Can a simple reflex agent with a randomized agent function outperform a simple reflex agent? Design such an agent (principle) and measure its performance on several environments (if you have time).    
3. Can you design an environment in which your randomized agent will perform poorly? Show your results.
"""

# ╔═╡ Cell order:
# ╠═88af7dc6-70fd-11eb-3af2-f7a194340290
# ╠═bb6ed424-70fc-11eb-2a2a-ff0556bb7cda
# ╠═9fd46218-70fe-11eb-1ddd-132f13e0611c
# ╟─ab564aec-70ff-11eb-0d36-cd95fb8def23
# ╟─f2915910-70ff-11eb-2f60-e9963db3aaeb
