### A Pluto.jl notebook ###
# v0.18.4

using Markdown
using InteractiveUtils

# ╔═╡ 78b36c68-895b-11eb-3754-2f78ee415f6c
begin
	# dependencies
	
	using Pkg
	Pkg.activate(joinpath(@__DIR__,".."))
	
	using PlutoUI
	using Plots

	using BenchmarkTools
	TableOfContents()
end

# ╔═╡ 9a85bb2e-e3f7-48e8-96ab-5f18840cb27f
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

# ╔═╡ 7a65ebb8-8958-11eb-3c1f-cf1daed9200e
md"""
# Adversarial search

"""

# ╔═╡ c7cf0a38-acb3-43b0-a6a8-0f03f7130e18
md"""
## Quick questions
* What is adversarial search?
* What is a solution in the context of adversarial search?
* Describe the difference between zero-sum and general games
* What problem does depth limited search solve? What is the impact on the evaluation function?
* How can pruning be used in adversarial search?
"""

# ╔═╡ 88a95bfa-895b-11eb-2336-b3e894079c4f
md"""
## Small applications

To be able to represent a game we need the following:
* ``S_0``: The initial state, which specifies how the game is set up at the start.
* ``\texttt{player}(s)``: Defines which player has the move in a state.
* ``\texttt{actions}(s)``: Returns the set of legal moves in a state.
* ``\texttt{result}(s,a)``: The transition model, which defines the result of a move.
* ``\texttt{terminal-test}(s)``: A terminal test, which is true when the game is over and false otherwise.
* ``\texttt{utility}(s, p)``: A utility function (also called an objective function or payoff function), defines the final numeric value for a game that ends in terminal state s for a player p
* ``\texttt{display}(g)``: Display a game in way that is easy to interpret for us humans (optional).
### Pursuit game (Ex 5.2/3)

Suppose two friends live in different cities on a map. Imagine that due to a contagious virus, one of the friends wants to avoid the other. Let's call the two persons $P$, the pursuer and $E$, the evader. Suppose that the players take turns moving. The game ends only when the players are on the same node; the terminal payoff to the pursuer is minus the total time taken (expressed in number of turns). The evader "wins" by never losing. An example is shown below (P moves first):
"""

# ╔═╡ 0a3f61fa-895c-11eb-3fae-f3cef63c52c4
PlutoUI.LocalResource("./img/adversarialmap.png", :width => 500, :align => "middle")

# ╔═╡ 6783207e-895a-11eb-2d87-bf236e515f97
md"""
A partial partial game tree for this map is shown in the image below. Each node is labeled by its position xy, e.g. bd corresponds with player $P$ on location b and player $E$ on location d. This is the starting position on the image above. 

"""


# ╔═╡ 615c38a0-895c-11eb-26bf-2f78b04eed50
PlutoUI.LocalResource("./img/adversarialtree.png", :width => 500, :align => "middle")

# ╔═╡ 544bdd62-895c-11eb-14fa-f15a675494f8
md"""
1. Formule the game in the adversial search framework.
1. Copy the game tree and mark the values of the terminal nodes (point of view of the pursuer).
2. Next to each internal node, write the strongest fact you can infer about its value (a number, one or more inequalities such as “≥ 14”, or a “?”).
3. Beneath each question mark, write the name of the node reached by that branch.
4. Explain how a bound on the value of the nodes in (3) can be derived from consideration of shortest-path lengths on the map, and derive such bounds for these nodes. Remember the cost to get to each leaf as well as the cost to solve it.
5. Now suppose that the tree as given, with the leaf bounds from (4), is evaluated from left to right. Circle those “?” nodes that would not need to be expanded further, given the bounds from part (4), and cross out those that need not be considered at all.
6. Can you claim anything in general about who wins the game on a map that is a tree? (Suppose that the tree is finite)
"""

# ╔═╡ 4dac7eea-895d-11eb-1ece-7bbfdf60aa4e
md"""
### Traversal game (Ex 5.8)
Consider the game illustrated below:


$(PlutoUI.LocalResource("./img/game2.png", :width => 300, :align => "middle"))
"""

# ╔═╡ b1e4afa4-895d-11eb-3f54-a32c30da203c
md"""
Player A moves first. The two players take turns moving, and each player must move his token to an open adjacent space in either direction. If the opponent occupies an adjacent space, then a player may jump over the opponent to the next open space if any. (For example, if A is on 3 and B is on 2, then A may move back to 1.) The game ends when one player reaches the opposite end of the board. If player A reaches space 4 first, then the value of the game to A is +1; if player B reaches space 1 first, then the value of the game to A is −1.

1. Formulate the different components we need in the adversarial search framework
1. Draw the complete game tree, using the following conventions:
    * Write each state as $(s_A, s_B )$, where $s_A$ and $s_B$ denote the token locations.
    * Put each terminal state in a square box and write its game value in a circle.
    * Put loop states (states that already appear on the path to the root) in double square boxes. Since their value is unclear, annotate each with a “?” in a circle.
2. Now mark each node with its backed-up minimax value (also in a circle). Explain how you handle the “?” values and why. You should provide small reasoning.
3. Explain why the standard minimax algorithm would fail on this game tree and briefly sketch how you might fix it, drawing on your answer to (2). Does your modified algorithm give optimal decisions for all games with loops?
"""

# ╔═╡ 9c62b3f8-bc06-4e4d-91a4-33c13c4ed6f4
md"""
### A-B pruning
On the illustration below, you see the state of a two-player, zero-sum game with the values of the terminal states. Which branches won’t be explored when using α − β pruning? Suppose nodes are evaluated from the left to the right.

$(PlutoUI.LocalResource("./img/abtree.png", :width => 300, :align => "middle"))

"""

# ╔═╡ 8ae142e6-8967-11eb-32fb-7f214a71d8b5
md"""
## Implementations 
### Minimax
Below you have an implementation of the minimax algorithm.
"""

# ╔═╡ 90cf8be8-4c49-497f-9f08-d52a6b962a2b
begin
	abstract type AbstractGame end
	abstract type AbstractGameState end
end

# ╔═╡ 1c7a90c2-af52-4ac3-944a-dbee27c8ceb7
md"""
### A-B pruning
Below you can find an implementation for a-b pruning
"""

# ╔═╡ 1d5c0a6d-800a-4331-a53d-022da618c1d7
begin
	#=
	abstract type AbstractGame end
	abstract type AbstractGameState end
	
	"""
		my_minimax_decision(state::S, game::G, depth::Int, maxplayer::Bool)

	Minimax implementation. Given all the possible actions of the current state of the game, returns the one that maximizes the utility for the current player. This is done by searching recursively through the moves all the way to the terminal states. Note that the number of players is not defined beforehand, so this method can be used for 
	"""
	function my_minimax_decision(game, state, depth::Int) 
		player = to_move(game, state) # number of the player who's turn it is.
		@info "determining best move for player $(player)"
		val, move = max_val(game, state, depth)

		return val, move
	end

	"""
		max_val(game::G, state::S, depth::Int, player::Int)

	Find maximal utility for current player
	"""
	function max_val(game, state, depth::Int) 
		if iszero(depth) || terminal_test(game, state)
			return utility(game, state), nothing
		end

		v, move = -Inf, nothing
		for action in actions(game, state)
			v2, a2 = max_val(game, result(game, state, action), depth-1)
			if v2 > v
				v, mode = v2, a2
			end
		end
		return v, move
	end

	"""
    	TicTacToeState

	Representation of a TicTacToe state.
	
	# Arguments
	- `turn::String`: current player ("X" or "O")
	- `utility::Vector{Int}`: utility for all players# +1/-1/0 (in case of a draw)
	- `board::Dict`: store game evolution (e.g. (1,3) => "X")
	- `board::Vector{Tuple{Int64,Int64}}`: holds all possible moves
	"""
	struct TicTacToeState <: AbstractGameState
		player::Int
		utility::Vector
		board::Dict
		moves::Vector{Tuple{Int64,Int64}}
	end

	"""
    	TicTacToeGame

	Implementation of the Tic-tac-toe game. 
		
	# Arguments
	- `initial::TicTacToeState`: initial state
	- `h::Int64`: number of horizontal rows
	- `v::Int64`: number of vertical rows
	- `k::Int64`: number of aligned values required to win
	"""
	struct TicTacToeGame <: AbstractGame
		initial::TicTacToeState
		h::Int64
		v::Int64
		k::Int64
	end
	
	"""
		TicTacToeGame(initial::TicTacToeState)
	
	From an initial state generate a 3x3 game.
	"""
	function TicTacToeGame(initial::TicTacToeState)
		return TicTacToeGame(initial, 3, 3, 3)
	end

	"""
		TicTacToeGame()
	
	Generate a 3x3 empty TicTacToe game with player "X" starting
	"""
	function TicTacToeGame()
		init = TicTacToeState(1, [-Inf; -Inf], Dict(), collect((x, y) for x in 1:3 for y in 1:3))
		return TicTacToeGame(init, 3, 3, 3)
	end

	"""
    	actions(game, state)

	Retrieve possible actions from the current state of the game.
	"""
	function actions(game::TicTacToeGame, state::TicTacToeState)
		return state.moves
	end
	
	"""
    	result(game, state, move)

	Return the state of a game after applying a move.
	"""
	function result(game::TicTacToeGame, state::TicTacToeState, move::Tuple{Int64,Int64})
		# only allowed moves
		if move ∉ state.moves
			@warn "Illegal move"
			return state
		end
		
		# copy board in order not to modify previous state & store move
		board = copy(state.board)
		board[move] = state.player
		
		# remaining moves
		moves = filter(x-> !isequal(x, move), state.moves)
		
		return TicTacToeState(state.player == 1 ? 2 : 1, # next player
   							  compute_utility(game, board, move, state.player), # update utility vector
							  board, # update board
							  moves) # update remaining moves

	end

	=#
end

# ╔═╡ 6a252a9e-895e-11eb-0410-29ab7456d6f9
md"""
## Application: TicTacToe
We will implement the minimax algorithm and apply it on a game of tic-tac-toe.Before going into the details of the implementation, we need to think about what we need and how we represent a game. Below you find an implementation of tic-tac-toe. 
"""

# ╔═╡ 5c0a3b24-895f-11eb-33aa-8d74ee1eaacd
begin	
	"""
    	TicTacToeState

	Representation of a TicTacToe state.
	
	# Arguments
	- `turn::String`: current player ("X" or "O")
	- `utility::Int64`: +1/-1/0 (in case of a draw)
	- `board::Dict`: store game evolution (e.g. (1,3) => "X")
	- `board::Vector{Tuple{Int64,Int64}}`: holds all possible moves
	"""
	struct TicTacToeState <: AbstractGameState
		turn::String
		utility::Int64
		board::Dict
		moves::Vector{Tuple{Int64,Int64}}
	end
	
	"""
    	TicTacToeGame

	Implementation of the Tic-tac-toe game. 
		
	# Arguments
	- `initial::TicTacToeState`: initial state
	- `h::Int64`: number of horizontal rows
	- `v::Int64`: number of vertical rows
	- `k::Int64`: number of aligned values required to win
	"""
	struct TicTacToeGame <: AbstractGame
		initial::TicTacToeState
		h::Int64
		v::Int64
		k::Int64
	end
	
	"""
		TicTacToeGame(initial::TicTacToeState)
	
	From an initial state generate a 3x3 game.
	"""
	function TicTacToeGame(initial::TicTacToeState)
		return TicTacToeGame(initial, 3, 3, 3)
	end

	"""
		TicTacToeGame()
	
	Generate a 3x3 empty TicTacToe game with player "X" starting
	"""
	function TicTacToeGame()
		init = TicTacToeState("X", 0, Dict(), collect((x, y) for x in 1:3 for y in 1:3))
		return TicTacToeGame(init, 3, 3, 3)
	end

	"""
    	actions(game, state)

	Retrieve possible actions from the current state of the game.
	"""
	function actions(game::TicTacToeGame, state::TicTacToeState)
		return state.moves
	end
	
	"""
    	result(game, state, move)

	Return the state of a game after applying a move.
	"""
	function result(game::TicTacToeGame, state::TicTacToeState, move::Tuple{Int64,Int64})
		# only allowed moves
		if move ∉ state.moves
			@warn "Illegal move"
			return state
		end
		
		# copy board in order not to modify previous state & store move
		board = copy(state.board)
		board[move] = state.turn
		
		# remaining moves
		moves = filter(x-> !isequal(x, move), state.moves)
		
		return TicTacToeState(state.turn == "X" ? "O" : "X", 
   							  compute_utility(game, board, move, state.turn),
							  board,
							  moves)

	end
	
	"""
    	utility(game, state, player::String)

	For a game, quantify the utility of a state given a player's turn 
	"""
	function utility(game::TicTacToeGame, state::TicTacToeState, player::String)
    	return player == "X" ? state.utility : -state.utility
	end
	
	"""
    	terminal_test(game, state)
	
	For a game, check if it's finished
	"""
	function terminal_test(game::TicTacToeGame, state::TicTacToeState)
		return (state.utility != 0) || (length(state.moves) == 0)
	end
	
	"""
    	to_move(game, state)

	Determine whose round it is for a game in a state.
	"""
	function to_move(game::TicTacToeGame, state::TicTacToeState)
    	return state.turn
	end
	
	"""
    	compute_utility(game, board, move, player)

	Compute the utility for a terminal state, dealing with all possible cases (rows, columns, diagonals) 
	"""
	function compute_utility(game::TicTacToeGame, board::Dict, 
							 move::Tuple{Int64, Int64}, player::String)
		# if three in a row is found, return matching utility value
		if k_in_row(game, board, move, player, (0, 1)) ||  # check rows
		   k_in_row(game, board, move, player, (1, 0)) ||  # check columns
		   k_in_row(game, board, move, player, (1, -1))||  # check main diagonal
		   k_in_row(game, board, move, player, (1, 1))     # check other diagonal

			return player == "X" ? 1 : -1
		else
			return 0
		end
	end
	
	"""
    	k_in_row(game, board, move, player, delta)

	Supporting function for `compute_utility`. Determine if a player obtains three in 	  a row after applying a move on the board. Checks in a direction determined by   
	delta.
	"""
	function k_in_row(game::TicTacToeGame, board::Dict, move::Tuple{Int64, Int64}, 						  player::String, delta::Tuple{Int64, Int64})
		(delta_x, delta_y) = delta
		(x, y) = move
		n = 0
		# go from the current move in one direction (adding delta_x, delta_y)
		while get(board, (x,y), nothing) == player
			n = n + 1;
			x = x + delta_x
			y = y + delta_y
		end
		# reset to current position 
		(x, y) = move
		# go from current move to the other direction (subtracting delta_x, delta_y)
		while get(board, (x,y), nothing) == player
			n = n + 1
			x = x - delta_x
			y = y - delta_y
		end
		# remove the duplicate check on the position of the move
		n = n - 1;  
		return n >= game.k
	end
	
	"""
    	display(game, state)

	Display the TicTacToe game in a human-readable way.
	"""
	function Base.display(game::TicTacToeGame, state::TicTacToeState)
		io = IOBuffer()	
		print(io, """Current game layout (X's score: $(state.utility), current player: $(state.turn)):\n\n""")
		for x in 1:game.h
			for y in 1:game.v
				print(io, get(state.board, (x, y), "."))
			end
			print(io, "\n")
    	end
		
		return println(stdout,String(take!(io)))
	end

	nothing
	
end

# ╔═╡ 1e045506-8966-11eb-2c2f-f39e9d812b3b
md"""
### Illustration of a small game
"""

# ╔═╡ 38c21f1c-8967-11eb-32ed-8bc5633cefd0
md"""
Note: for this illustration we do not check if a player wins during the (random) game,  we just fill the entire board to illustrate the game is working as intended.
"""

# ╔═╡ abb41398-8969-11eb-29d0-05fe2438b3a8
md"""
### Using minimax
Below we have a game that is near completion (only two cases left). Player $X$ has the turn. It is clear that the best possible move is to choose $(3,3)$, as this will lead to a victory for $X$. We see that the minimax algorithm correctly identifies this move. 
"""

# ╔═╡ a65acd26-896c-11eb-21f0-3b8479d17e1c
md"""
### Playing a game
Try to realise the following:

* have two "intelligent" agents play against eachother
* have one "intelligent" agent play against a "dumb" agent i.e. an agent that makes random moves

What are the outcomes? Is this to be expected?
"""

# ╔═╡ bb6e0a6a-896d-11eb-13da-adbd86f43b93
md"""
### Using A-B pruning
In the previous application you have used and the minimax algorithm. Below we use the  $\alpha\text{-}\beta$ pruning algorithm (still exploring the game tree to the bottom). We use it in the same setting as before (i.e. TicTacToe) and compare its performance with the standard minimax algorithm.
"""

# ╔═╡ af046f97-5bc9-4cce-8d27-9b510308b3e6
md"""
# Adding uncertainty
## Small applications
### Chance tree (Ex 5.17)
"""


# ╔═╡ b2e7ac62-614c-4b56-ab13-3b3826f4265e
md"""
This question considers pruning in games with chance nodes. Consider the complete game tree for a trivial game: 
"""

# ╔═╡ ea98ae70-2c43-439e-a553-7573b321966e
PlutoUI.LocalResource("./img/chancetree.png", :width=>400)

# ╔═╡ 9dffa40c-8de0-4446-b3d2-551a112f29e2
md"""
Assume that the leaf nodes are to be evaluated in left-to-right order, and that before a leaf node is evaluated, we know nothing about its value, i.e. the range of possible values is $-\infty$ to $-\infty$.

1. Copy the figure, mark the value of all the internal nodes, and indicate the best move at the root with an arrow.
2. Given the values of the first six leaves, do we need to evaluate the seventh and eighth leaves? 
3. Given the values of the first seven leaves, do we need to evaluate the eighth leaf? Explain your answers.
4. Suppose the leaf node values are known to lie between –2 and 2 inclusive. After the first two leaves are evaluated, what is the value range for the left-hand chance node?
5. Circle all the leaves that need not be evaluated under the assumption in (4). Explain your answers.
"""

# ╔═╡ b1ebdb99-6ce8-4de8-9f15-e60201a7259b
md"""
### Max tree (5.21) 
In the following, a “max” tree consists only of max nodes, whereas an “expectimax” tree consists of a max node at the root with alternating layers of chance and max nodes. At chance nodes, all outcome probabilities are nonzero. The goal is to find the value of the root with a bounded-depth search. An illustration is shown below.

"""

# ╔═╡ ab17b5a7-0969-4601-ab5e-a65c235e6bc3
md"""
Max tree             |  Expectimax tree
:-------------------------:|:-------------------------:
$(PlutoUI.LocalResource("./img/maxtree.png",:width=>200)) | $(PlutoUI.LocalResource("./img/expectimaxtree.png",:width=>200))

1. Assuming that leaf values are finite but unbounded, is pruning (as in alpha–beta) ever possible in a max tree? Give an example, or explain why not.
2. Is pruning ever possible in an expectimax tree under the same conditions? Give an example, or explain why not.
3. If leaf values are constrained to be non-negative, is pruning ever possible in a max tree? Give an example, or explain why not.
4. If leaf values are constrained to be non-negative, is pruning ever possible in an expectimax tree? Give an example, or explain why not.
5. If leaf values are constrained to be in the range $[0, 1]$, is pruning ever possible in a max tree? Give an example, or explain why not.
6. If leaf values are constrained to be in the range $[0, 1]$, is pruning ever possible in an expectimax tree? Give an example (qualitatively different from your example in (5), if any) or explain why not.
7. Consider the outcomes of a chance node in an expectimax tree. Which of the following evaluation orders is most likely to yield pruning opportunities: (i) Lowest probability first; (ii) Highest probability first; (iii) Doesn’t make any difference?
"""


# ╔═╡ 782317ee-6d90-4e00-8609-23bb7102382f
md"""
### Gambling game
Consider the following game:
* each  player gets to play $n$ times. At each round a player can decide if he wants to add one or two to his score. 
* after each player, a coin is flipped. If it's heads, one of your own points goes to the other player, if not, nothing changes. Assume that the coin is not biased and you have a 50/50 chance between heads and tails.
* the value in a leaf node is given by the difference between max and min.

Determine the best move(s) for a game of depth 1. Do this for both `Min` and `Max` as first player.
"""

# ╔═╡ 67c91d61-d68a-46f3-b33a-629adba5716f
md"""
## Expectimax implementation 
Below you can find an implementation for the expectimax algorithm.
"""

# ╔═╡ 4ca46ca6-ed67-4143-aabe-be8509e7173a
md"""
## Application - Gambling game
Consider the gambling game from earlier. Below you can find an implementation of the game. Use expectimax in the following cases:
* Determine the best move(s) for a game of depth 1. Do this for both `Min` and `Max` as first player.
* Consider a game of limited length (``n\le5``). Look at the distrubution of the outcomes of multiple games when playing against:
   1. a random player
   2. an optimal player. 
  Discuss your results.
"""

# ╔═╡ 9aeb26c2-5949-40bf-86e0-f6d6353c5b16
begin
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

	nothing
end

# ╔═╡ ab40ff71-e21b-47ff-baf6-af76804f01fd
begin

	
	
	"""
    	minimax_decision(state, game)
	
	Given all the possible actions of the current state of the game, returns the one that maximizes the utility. This is done by searching through the moves all the way to the terminal states.
	
	Makes us of a helper function inside (called `foo`) that determines the utility of a specific action).
	
	Note: argmax returns the index of the largest value.
	"""
	function minimax_decision(state::M, game::T) where {M <:AbstractGameState, T <: AbstractGame}
		# detemermine what player needs to make a move
		player = to_move(game, state);

		# supporting function for argmax
		function foo(action::Tuple{Int64,Int64}; 
					 relevant_game::AbstractGame=game, 
					 relevant_player::String=player, 
				     relevant_state::M=state)
			return minimax_min_value(relevant_game, 
									 relevant_player, 
									 result(relevant_game, relevant_state, action));
		end

		bestactionindex = argmax([foo(action) for action in actions(game, state)])
		
		return actions(game, state)[bestactionindex]
	end;
	
	
	"""
		minimax_min_value(game, player, state)
	
	return the minimal utility value of all possible actions for a game in a state
	"""
	function minimax_min_value(game::T, player::String, state::M) where {T <: AbstractGame, M <: AbstractGameState}
		if (terminal_test(game, state))
			return utility(game, state, player);
		end
			
		outcomes= [minimax_max_value(game, player, result(game, state, action)) for action in actions(game, state)]
		
		return minimum(vcat(Inf,outcomes))
	end
	
	
	"""
    	minimax_max_value(game, player, state)

	return the maximal utility value of all possible actions for a game in a state
	"""
	function minimax_max_value(game::T, player::String, state::M) where {T <: AbstractGame, M <: AbstractGameState}
		if (terminal_test(game, state))
			return utility(game, state, player)
		end
		
		outcomes = [minimax_min_value(game, player, result(game, state, action)) for action in actions(game, state)]

		return maximum(vcat(-Inf, outcomes))
	end
	nothing
end

# ╔═╡ c6a5b54e-8969-11eb-056e-9d4dd769f7c4
let
	G = TicTacToeGame()
	state = TicTacToeState("X",0, Dict((1,1)=>"X", (1,2)=>"O", (1,3)=>"X", 
									   (2,1)=>"O", (2,2)=>"X",
									   (3,1)=>"X",(3,2)=>"O"), 
						   Tuple{Int64,Int64}[(2, 3), (3, 3)])
	display(G,state)
	move = minimax_decision(state, G);
	println("Player $(state.turn)'s best possible move: $(move)")
end

# ╔═╡ ee38ca70-896d-11eb-1256-97a4b1b3f321
begin
	"""
 	   alphabeta_full_search(state, game)

	Given all the possible actions of the current state of the game, returns the one that maximizes the utility. This is done by searching through the moves all the way to the terminal states, making use of α-β pruning (Fig 5.7)
	
	Makes us of a helper function inside (called `foo`) that determines the utility of a specific action).
	"""
	function alphabeta_full_search(state::M, game::T) where {M <:AbstractGameState, T <: AbstractGame}
		# detemermine what player needs to make a move
		player = to_move(game, state);

		# supporting function for argmax
		function foo(action::Tuple{Int64,Int64}; 
					 relevant_game::AbstractGame=game, 
					 relevant_player::String=player, 
					 relevant_state::M=state)
			 return alphabeta_full_search_min_value(relevant_game, 
													relevant_player, 
						result(relevant_game, relevant_state, action), -Inf, Inf);
		end

		bestactionindex = argmax([foo(action) for action in actions(game, state)])
		return actions(game, state)[bestactionindex]
	end
	
	
	function alphabeta_full_search_max_value(game::T, 
											 player::String, 
											 state::M, 
											 α::Number, β::Number) where {T <: AbstractGame, M<:AbstractGameState}
		if (terminal_test(game, state))
			return utility(game, state, player)
		end
    	v = -Inf
    	for action in actions(game, state)
        	v = max(v, alphabeta_full_search_min_value(game, player, result(game, state, action), α, β))
        	if (v >= β)
            	return v
	        end
    	    α = max(α, v)
	    end
    	return v
	end

	function alphabeta_full_search_min_value(game::T, 
											 player::String, 
											 state::M, 
											 α::Number, β::Number) where {T <: AbstractGame, M<:AbstractGameState}
    	if (terminal_test(game, state))
			return utility(game, state, player)
		end
		
		v = Inf
		for action in actions(game, state)
			v = min(v, alphabeta_full_search_max_value(game, player, result(game, state, action), α, β))
			if (v <= α)
				return v
			end
			β = min(β, v)
		end
		return v
	end
	nothing
end

# ╔═╡ cd81ae40-8a22-11eb-04a3-1bb77668a99b
let
	# three remaining values
	G = TicTacToeGame()
	state = TicTacToeState("O",
						   0, 
						   Dict((1,1)=>"X", (1,2)=>"O", (2,1)=>"O", (2,2)=>"X",(3,1)=>"X",(3,2)=>"O"), 
						   Tuple{Int64,Int64}[(2, 3), (3, 3),(1,3)])
	display(G,state)
	@btime minimax_decision($state, $G)
	@btime alphabeta_full_search($state, $G)
end

# ╔═╡ 298bb4d2-8966-11eb-2d9e-e15e24682397
begin
	# Initialise the game and show it's layout
	G = TicTacToeGame()
	state = G.initial
	display(G, state)
	# make random moves
	for _ = 1:9
		global state = result(G, state, rand(actions(G, state)))
		display(G,state);println()
	end
end

# ╔═╡ 4c136544-8967-11eb-322c-952f52cbf62f
begin
	@show state
	nothing
end

# ╔═╡ 7711bc0c-8967-11eb-0fdf-e30e182b06b1
@assert terminal_test(G, state) # game over!

# ╔═╡ 15b8bc4a-8b08-11eb-1991-95df70d5d7ee
begin
	function playintelligent(n=10)
		for _ = 1:n
			G = TicTacToeGame()
			state = G.initial
			for _ in 1:9
				action = minimax_decision(state, G)
				state = result(G, state, action)
			end
			msg = ifelse(state.utility == 0, "it's a draw", state.utility == 1 ? "X has won" : "O has won")
			println("Game state: $(msg)")
		end
	end
	
	function playagainstrandom(n=10)
		for _ = 1:n
			G = TicTacToeGame()
			state = G.initial
			for _ in 1:9
				if state.turn == "X"
					action = minimax_decision(state, G)
					state = result(G, state, action)
				else
					action = rand(actions(G, state))
					state = result(G, state, action)
				end
			end
			msg = ifelse(state.utility == 0, "it's a draw", state.utility == 1 ? "X has won" : "O has won")
			println("Game state: $(msg)")
		end
	end
	nothing
end

# ╔═╡ a5c7dd32-8b14-11eb-197f-895819e95ff5
playagainstrandom()

# ╔═╡ 47465a38-24d9-483d-9951-295233b864b5
playintelligent()

# ╔═╡ 5a5f6743-e7d9-41a1-8c43-9c5acaf1236a
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
	nothing
end

# ╔═╡ 8c1e8e2a-45e9-45c8-8970-4402c37828bb
ProbGameState()

# ╔═╡ 5f2cf00a-1ee0-46e9-8d93-cde8edcbec92
ProbGame(n=1)

# ╔═╡ 225a80e5-a9e5-4580-aa6e-acf4b51cf2ae
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

# ╔═╡ c6c45e42-ec17-4896-86d9-7d8cd2e71662
let
	@info "Determine best move for game of depth 1"
	# game setup
	game = ProbGame(n=1)
	@show game
	# best move for player (and expected utility)
	bestmove = expectiminmaxchoice(game,game.initial)
	@show bestmove
end

# ╔═╡ 3c62ec56-66e6-42ed-9093-3a1ec4f2b8b0
let
	@info "illustration of a game of depth 1 starting with min"
	# game setup
	game = ProbGame(ProbGameState("min", (0,0), "", 0),n=1)
	@show game
	# best move for player (and expected utility)
	bestmove = expectiminmaxchoice(game,game.initial)
	@show bestmove
end

# ╔═╡ 41f62f77-6dc6-41f0-a52e-91dd641ad8e1


# ╔═╡ a5b76867-1e62-4e85-9f41-07ea99b52a8d
md"""
# Additional challenges
Consider the game [Connect 4 x 4](https://en.wikipedia.org/wiki/Connect_4x4). This game extends the classical connect 4 game from a two-player to a four-player game by adding an additional parallel grid.

Tasks:
* Come up with an adversarial game formulation for this game
* how would you need to modifiy the existing minimax algorithm to play this game?
* Implement this game and the minimax extension. 
"""

# ╔═╡ Cell order:
# ╟─9a85bb2e-e3f7-48e8-96ab-5f18840cb27f
# ╟─7a65ebb8-8958-11eb-3c1f-cf1daed9200e
# ╟─c7cf0a38-acb3-43b0-a6a8-0f03f7130e18
# ╟─78b36c68-895b-11eb-3754-2f78ee415f6c
# ╟─88a95bfa-895b-11eb-2336-b3e894079c4f
# ╟─0a3f61fa-895c-11eb-3fae-f3cef63c52c4
# ╟─6783207e-895a-11eb-2d87-bf236e515f97
# ╟─615c38a0-895c-11eb-26bf-2f78b04eed50
# ╟─544bdd62-895c-11eb-14fa-f15a675494f8
# ╟─4dac7eea-895d-11eb-1ece-7bbfdf60aa4e
# ╟─b1e4afa4-895d-11eb-3f54-a32c30da203c
# ╟─9c62b3f8-bc06-4e4d-91a4-33c13c4ed6f4
# ╟─8ae142e6-8967-11eb-32fb-7f214a71d8b5
# ╠═90cf8be8-4c49-497f-9f08-d52a6b962a2b
# ╠═ab40ff71-e21b-47ff-baf6-af76804f01fd
# ╟─1c7a90c2-af52-4ac3-944a-dbee27c8ceb7
# ╠═ee38ca70-896d-11eb-1256-97a4b1b3f321
# ╟─1d5c0a6d-800a-4331-a53d-022da618c1d7
# ╟─6a252a9e-895e-11eb-0410-29ab7456d6f9
# ╠═5c0a3b24-895f-11eb-33aa-8d74ee1eaacd
# ╟─1e045506-8966-11eb-2c2f-f39e9d812b3b
# ╠═298bb4d2-8966-11eb-2d9e-e15e24682397
# ╟─38c21f1c-8967-11eb-32ed-8bc5633cefd0
# ╠═4c136544-8967-11eb-322c-952f52cbf62f
# ╠═7711bc0c-8967-11eb-0fdf-e30e182b06b1
# ╟─abb41398-8969-11eb-29d0-05fe2438b3a8
# ╠═c6a5b54e-8969-11eb-056e-9d4dd769f7c4
# ╟─a65acd26-896c-11eb-21f0-3b8479d17e1c
# ╠═15b8bc4a-8b08-11eb-1991-95df70d5d7ee
# ╠═a5c7dd32-8b14-11eb-197f-895819e95ff5
# ╠═47465a38-24d9-483d-9951-295233b864b5
# ╟─bb6e0a6a-896d-11eb-13da-adbd86f43b93
# ╠═cd81ae40-8a22-11eb-04a3-1bb77668a99b
# ╟─af046f97-5bc9-4cce-8d27-9b510308b3e6
# ╟─b2e7ac62-614c-4b56-ab13-3b3826f4265e
# ╟─ea98ae70-2c43-439e-a553-7573b321966e
# ╟─9dffa40c-8de0-4446-b3d2-551a112f29e2
# ╟─b1ebdb99-6ce8-4de8-9f15-e60201a7259b
# ╟─ab17b5a7-0969-4601-ab5e-a65c235e6bc3
# ╟─782317ee-6d90-4e00-8609-23bb7102382f
# ╟─67c91d61-d68a-46f3-b33a-629adba5716f
# ╠═5a5f6743-e7d9-41a1-8c43-9c5acaf1236a
# ╟─4ca46ca6-ed67-4143-aabe-be8509e7173a
# ╠═9aeb26c2-5949-40bf-86e0-f6d6353c5b16
# ╠═8c1e8e2a-45e9-45c8-8970-4402c37828bb
# ╠═5f2cf00a-1ee0-46e9-8d93-cde8edcbec92
# ╠═225a80e5-a9e5-4580-aa6e-acf4b51cf2ae
# ╠═c6c45e42-ec17-4896-86d9-7d8cd2e71662
# ╠═3c62ec56-66e6-42ed-9093-3a1ec4f2b8b0
# ╠═41f62f77-6dc6-41f0-a52e-91dd641ad8e1
# ╟─a5b76867-1e62-4e85-9f41-07ea99b52a8d
