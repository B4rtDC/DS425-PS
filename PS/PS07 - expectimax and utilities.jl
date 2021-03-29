### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ f79f691c-8e49-11eb-0aff-714a3f5248d5
using PlutoUI, Distributions, Plots

# ╔═╡ 70426596-8e49-11eb-2455-a9818cad032b
md"""
# Expectimax
"""

# ╔═╡ 543cd594-8e4a-11eb-1101-2d188eda8b88
md"""
## Application 1 (5.16)
"""

# ╔═╡ 64fbc4f6-8e4a-11eb-36d8-29ff6f774100
PlutoUI.LocalResource("./img/chancetree.png", :width=>400)

# ╔═╡ e1dffc86-8e49-11eb-10b7-e795ee87a053
md"""
This question considers pruning in games with chance nodes. Consider the complete game tree for a trivial game shown above. 

Assume that the leaf nodes are to be evaluated in left-to-right order, and that before a leaf node is evaluated, we know nothing about its value, i.e. the range of possible values is $-\infty$ to $-\infty$.

1. Copy the figure, mark the value of all the internal nodes, and indicate the best move at the root with an arrow.
2. Given the values of the first six leaves, do we need to evaluate the seventh and eighth leaves? 
3. Given the values of the first seven leaves, do we need to evaluate the eighth leaf? Explain your answers.
4. Suppose the leaf node values are known to lie between –2 and 2 inclusive. After the first two leaves are evaluated, what is the value range for the left-hand chance node?
5. Circle all the leaves that need not be evaluated under the assumption in (4). Explain your answers."""

# ╔═╡ ed342f68-8e4a-11eb-0173-a3cd935b42eb
md"""
## Application 2 (5.21)

In the following, a “max” tree consists only of max nodes, whereas an “expectimax” tree consists of a max node at the root with alternating layers of chance and max nodes. At chance nodes, all outcome probabilities are nonzero. The goal is to find the value of the root with a bounded-depth search. An illustration is shown below.
"""

# ╔═╡ 96eb968e-8e4b-11eb-0d82-f7cf952a4b55
PlutoUI.LocalResource("./img/expectimaxtree.png",:width=>200)

# ╔═╡ 11d98910-8e4b-11eb-2097-69a6a67b97b0
begin
	PlutoUI.LocalResource("./img/maxtree.png",:width=>200)
end

# ╔═╡ a719f96a-8e4b-11eb-1077-8be6a992691a
md"""
1. Assuming that leaf values are finite but unbounded, is pruning (as in alpha–beta) ever possible in a max tree? Give an example, or explain why not.
2. Is pruning ever possible in an expectimax tree under the same conditions? Give an example, or explain why not.
3. If leaf values are constrained to be non-negative, is pruning ever possible in a max tree? Give an example, or explain why not.
4. If leaf values are constrained to be non-negative, is pruning ever possible in an expectimax tree? Give an example, or explain why not.
5. If leaf values are constrained to be in the range $[0, 1]$, is pruning ever possible in a max tree? Give an example, or explain why not.
6. If leaf values are constrained to be in the range $[0, 1]$, is pruning ever possible in an expectimax tree? Give an example (qualitatively different from your example in (5), if any) or explain why not.
7. Consider the outcomes of a chance node in an expectimax tree. Which of the following evaluation orders is most likely to yield pruning opportunities: (i) Lowest probability first; (ii) Highest probability first; (iii) Doesn’t make any difference?
"""

# ╔═╡ 282f47d0-8e4c-11eb-3e1e-d15a008161b3
md"""
## Application 3 - game
* each  player gets to play $n$ times. At each round a player can decide if he wants to add one or two to his score. 
* after each player, a coin is flipped. If it's heads, one of your own points goes to the other player, if not, nothing changes. Assume that the coin is not biased and you have a 50/50 chance between heads and tails.
* the value in a leaf node is given by the difference between max and min.

This game has been implemented following the layout you already know from previous sessions. Try to implement the expecitiminiax algorithm and use it on this game. For a game with limited length you can draw the game tree and reason on it.

"""

# ╔═╡ 387aaea8-8e4d-11eb-0abb-93c31193af2f
begin
	abstract type AbstractGame end 
	abstract type AbstractGameState end
	
	"""
		ProbGameState

	Representation of a probabilistic game state
	
	# Arguments
	`turn::String`: current player
	`score::Tuple{Int64,Int64}`: tuple holding the scores for Max and Min
	`prevplay::String`: previous 'human' player (i.e. not a chance node)
	`nturns::Int64`: number of turns that have been completed
	"""
	struct ProbGameState <: AbstractGameState
		turn::String
		score::Tuple{Int64,Int64}
		prevplay::String
		nturns::Int64
	end
	
	function Base.show(io::IO, s::ProbGameState)
    	print(io, "ProbGameState\n-------------\nCurrent player: $(s.turn)\nscore: $(s.score)\nPrevious 'human' player: $(s.prevplay)\nNumber of turns played: $(round(Int,s.nturns/2))")
	end
	
	function ProbGameState()
		return ProbGameState("max",(0,0),"",0)
	end
	
	"""
		ProbGame

	Representation of a probabilistic game. 
	
	# Notes 
	Game starts with max
	
	# Arguments
	`initial::ProbGameState`: initial state
	`n::Int64`: number of turns for each player
	"""
	struct ProbGame <: AbstractGame
		initial::ProbGameState
		n::Int64
	end

	function ProbGame(initial::ProbGameState=ProbGameState(); n=3)
		return ProbGame(initial, n)
	end

	function Base.show(io::IO, g::ProbGame)
		print(io, "ProbGame with a depth of $(g.n)")
	end
	
	"""
		actions(game, state)

	Retrieve possible actions from the current state of the game.
	"""
	function actions(game::ProbGame, state::ProbGameState)
		if state.turn == "max"
			return [(1,0), (2,0)]
		elseif state.turn == "min"
			return [(0,1), (0,2)]
		elseif state.turn == "chance"
			if state.prevplay == "max"
				return [(-1,1), (0,0)]
			else
				return [(1,-1), (0,0)]
			end
		end
	end
	
	"""
		result(game, state, move)

	Given a game, its state and the move that is played, return the new state.
	"""
	function result(game::ProbGame, state::ProbGameState, move::Tuple{Int64,Int64})
		prevplay = state.prevplay
		nturns = state.nturns
		# determine next player
		if state.turn ≠ "chance"
			nturns += 1
			nextplayer = "chance"
			if state.turn == "max"
				prevplay = "max"
			else
				prevplay = "min"
			end 
		else
			if state.turn == "chance" && state.prevplay == "max"
				nextplayer = "min"
			else
				nextplayer = "max"
			end
		end
		# determine new score
		newscore = state.score .+ move

		return ProbGameState(nextplayer, newscore, prevplay, nturns)
	end
	
	"""
		terminal_test(game, state)

	Verify a if game has ended
	"""
	function terminal_test(game::ProbGame, state::ProbGameState)
		return game.n == state.nturns/2
	end
	
	"""
		utility(game, state)
	
	Returns the utility for a player of a game in a state
	"""
	function utility(game::ProbGame, state::ProbGameState)
		r = state.score[1] - state.score[2]
		if state.prevplay == "max"
			return -r
		elseif state.prevplay == "min"
			return r
		end    
	end
	
	"""
		to_move(game, state)

	Determine whos turn it is.
	"""
	function to_move(game::ProbGame, state::ProbGameState)
		return state.turn
	end
	
	"""
		prob(move::Tuple{Int64,Int64})
	
	Probablity of a given move (all equal to 0.5, cf. problem description)
	"""
	function prob(move::Tuple{Int64,Int64})
		return 0.5
	end
end

# ╔═╡ d3b39b3c-8e4d-11eb-123f-0307f1775d20
ProbGameState()

# ╔═╡ 4e82fd7a-8e4f-11eb-1e96-4daacfbfa65d
ProbGame(n=1)

# ╔═╡ 53e38136-8e4f-11eb-046f-95806459c80e
## illustration of a small game
let
	@info "Play a small game with random actions"
	# game setup
	game = ProbGame(n=1)
	state = game.initial
	# keep playing until the game is finished
	while !terminal_test(game,state) 
		@show state
		action = rand(actions(game, state))
		@show action
    	state = result(game, state, action)
	end
	
	@show state
	@show utility(game, state)
end

# ╔═╡ 33b23a08-8e52-11eb-2c06-dfbd8939d670
md"""
Below you can find an implementation of the EXPECTIMINIMAX algorithm
"""

# ╔═╡ 9e1a905c-8e52-11eb-0aa0-89aaf3701a29
begin
	"""
		expectiminmaxchoice(game, state)
	
	Determine the best action (move) for a stochastic game in a state.
	"""
	function expectiminmaxchoice(game::T, state::M) where {T <: AbstractGame, M <: AbstractGameState}
		# list of tuples (move, expexted value)
		movescores = [(move, expectiminmaxvalue(game, result(game, state, move))) for move in actions(game, state)]

		if state.turn == "max"
			return sort(movescores, by=x->x[2], rev=true)[1][1]
		else
			return sort(movescores, by=x->x[2], rev=false)[1][1]
		end
	end
	
	"""
		expectiminmaxvalue(game, state)
	
	Determine the expectiminmax value for a game in a state.
	"""
	function expectiminmaxvalue(game::T, state::M) where {T <: AbstractGame, M <: AbstractGameState}
		player = state.turn

		# If we are in a terminal state, we return the utility of the player
		if (terminal_test(game, state))
			r = utility(game, state)
			return r
		end

		# otherwise, we keep exploring the three
		if player == "max"
			return maximum([expectiminmaxvalue(game, result(game, state, move)) for move in actions(game, state)])
		elseif player == "min"
			return minimum([expectiminmaxvalue(game, result(game, state, move)) for move in actions(game, state)])
		elseif player == "chance"
			r = sum([prob(move)*expectiminmaxvalue(game, result(game, state, move)) for move in actions(game,state)])
			return r
		end
	end
end

# ╔═╡ 633d00e0-8e53-11eb-3f5c-55bb14c56012
let
	@info "Determine best move for game of depth 1"
	# game setup
	game = ProbGame(n=1)
	@show game
	# best move for player (and expected utility)
	bestmove = expectiminmaxchoice(game,game.initial)
	@show bestmove
end

# ╔═╡ b5ce4ea0-8e54-11eb-36f1-bd256cc6cef7
let
	@info "illustration of a game of depth 1 starting with min"
	# game setup
	game = ProbGame(ProbGameState("min", (0,0), "", 0),n=1)
	@show game
	# best move for player (and expected utility)
	bestmove = expectiminmaxchoice(game,game.initial)
	@show bestmove
end

# ╔═╡ e5168334-8e55-11eb-32aa-056184ae94ed
md"""
# Making simple decision
*Note*: In section 16.3.4, a lot is said about human behavior. One of the authors quoted is Daniel Kahneman. He wrote an entire book about how easy it is to trick the mind (e.g. by framing, anchoring) and how hard is for us to make unbiased decisions. If you are interested, the book is called ["Thinking, Fast and Slow"](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow) and is definitely worth a read.

## Application 1 (16.3)
In 1713, Nicolas Bernoulli stated a puzzle, now called the St. Petersburg paradox, which works as follows. You have the opportunity to play a game in which a fair coin is tossed repeatedly until it comes up heads. If the first heads appears on the n$^{th}$ toss, you win $2^n$ dollars.
1. Show that the expected monetary value of this game is infinite.
2. How much would you, personally, pay to play the game?
3. Nicolas’s cousin Daniel Bernoulli resolved the apparent paradox in 1738 by suggesting that the utility of money, $U(S_n)$, is measured on a logarithmic scale: $U(S_n) = a \log_2(m)+b$, where $S_n$ is the state of having $m$ dollars. What is the expected utility of the game under this assumption? Suppose you use the entirety of your capital to play this game.
4. How would you describe a rational player in this case? What is the maximum amount that would be rational to pay to play the game, assuming that (1) one's initial wealth is $k$ (2) one uses a quantity $c$ to play the game (3) a single game costs $1$ dollar?
"""

# ╔═╡ 6b0aead6-8e57-11eb-0ed3-f728f1bf72dd


# ╔═╡ 194dde60-8e57-11eb-0719-f18672c2fc48
md"""
## Application 2 (16.10) - Expected utility and post-decision disappointment
Let $X_1,\dots , X_k$ be continuous variables, independently distributed according to the same probability density function $f_X(x)$. Show that the density function for $\max \{ X_1,\dots,X_k \} $ is given by $kf_X(x)(F_X(x))^{k−1}$, where $F_X(x)$ is the cumulative distribution for $f_X$.

Some hints:
* What do you remember from independent events in terms of probability?
* What is the relation between $F_X(x)$ and $f_X(x)$?

Once you find the theoretical distribution, run a simulation to illustrate the bias that is introduced by using estimates of the expected utilty (cf. probability & statistics + modelling & simulation course).
1. try to experimentally reproduce figure 16.3.
2. use another distribution, look at the effect.
"""

# ╔═╡ 3e8fa6b2-8e58-11eb-1039-a9f966b12266
begin
	"""
		makedata(d::Distribution, k::Int64, n=1000)

	Generate a sample of length n from a distribution d for the maximum of k estimates

	"""
	function makedata(d::Distribution, k::Int64, n=1000)
		return maximum.([rand(d,k) for _ in 1:n])
	end

	"""
		makeplot(dist::Distribution, kvals::Array{Int64,1}=[3, 10, 30, 100], n=1000)

	Generates an plot of the distribution of the maximum of k utility estimates (for k in kval).

	Both the experimental and the theoretical distribution are shown.

	"""
	function makeplot(dist::Distribution, kvals::Array{Int64,1}=[3, 10, 30, 100], n=1000)
		f(x) = pdf(dist,x)
		F(x) = cdf(dist,x)
		Plots.histogram()
		for k in kvals
			s = makedata(dist,k,1000)
			x = range(minimum(s),maximum(s),length=500)
			Plots.histogram!(s, label="k = $(k)", alpha=0.6, normalize=:pdf)
			Plots.plot!(x,k*f.(x) .* F.(x) .^(k-1), label="", linecolor=:black, lw=4, ls=:dash)
		end
		Plots.xlabel!("x")
		Plots.ylabel!("experimental PDF")
		Plots.title!("Distribution of maxima for\n $(dist)")
	end
end

# ╔═╡ 64859af2-8e58-11eb-24c3-93dcbcf1901d
makeplot(Normal())

# ╔═╡ 6af7da96-8e58-11eb-376d-7b587a5c31f0
makeplot(Exponential(10))

# ╔═╡ Cell order:
# ╠═f79f691c-8e49-11eb-0aff-714a3f5248d5
# ╟─70426596-8e49-11eb-2455-a9818cad032b
# ╟─543cd594-8e4a-11eb-1101-2d188eda8b88
# ╟─64fbc4f6-8e4a-11eb-36d8-29ff6f774100
# ╟─e1dffc86-8e49-11eb-10b7-e795ee87a053
# ╟─ed342f68-8e4a-11eb-0173-a3cd935b42eb
# ╟─96eb968e-8e4b-11eb-0d82-f7cf952a4b55
# ╟─11d98910-8e4b-11eb-2097-69a6a67b97b0
# ╟─a719f96a-8e4b-11eb-1077-8be6a992691a
# ╟─282f47d0-8e4c-11eb-3e1e-d15a008161b3
# ╠═387aaea8-8e4d-11eb-0abb-93c31193af2f
# ╠═d3b39b3c-8e4d-11eb-123f-0307f1775d20
# ╠═4e82fd7a-8e4f-11eb-1e96-4daacfbfa65d
# ╠═53e38136-8e4f-11eb-046f-95806459c80e
# ╟─33b23a08-8e52-11eb-2c06-dfbd8939d670
# ╠═9e1a905c-8e52-11eb-0aa0-89aaf3701a29
# ╠═633d00e0-8e53-11eb-3f5c-55bb14c56012
# ╠═b5ce4ea0-8e54-11eb-36f1-bd256cc6cef7
# ╟─e5168334-8e55-11eb-32aa-056184ae94ed
# ╠═6b0aead6-8e57-11eb-0ed3-f728f1bf72dd
# ╟─194dde60-8e57-11eb-0719-f18672c2fc48
# ╟─3e8fa6b2-8e58-11eb-1039-a9f966b12266
# ╠═64859af2-8e58-11eb-24c3-93dcbcf1901d
# ╠═6af7da96-8e58-11eb-376d-7b587a5c31f0
