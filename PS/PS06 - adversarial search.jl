### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 78b36c68-895b-11eb-3754-2f78ee415f6c
begin
	using PlutoUI
end

# ╔═╡ 7a65ebb8-8958-11eb-3c1f-cf1daed9200e
md"""
# Adversarial search
"""

# ╔═╡ 88a95bfa-895b-11eb-2336-b3e894079c4f
md"""
## Application 1 (Ex 5.3)

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

1. Copy the game tree and mark the values of the terminal nodes (point of view of the pursuer).
2. Next to each internal node, write the strongest fact you can infer about its value (a number, one or more inequalities such as “≥ 14”, or a “?”).
3. Beneath each question mark, write the name of the node reached by that branch.
4. Explain how a bound on the value of the nodes in (3) can be derived from consideration of shortest-path lengths on the map, and derive such bounds for these nodes. Remember the cost to get to each leaf as well as the cost to solve it.
5. Now suppose that the tree as given, with the leaf bounds from (4), is evaluated from left to right. Circle those “?” nodes that would not need to be expanded further, given the bounds from part (4), and cross out those that need not be considered at all.
6. Can you claim anything in general about who wins the game on a map that is a tree? (Suppose that the tree is finite)
"""

# ╔═╡ 4dac7eea-895d-11eb-1ece-7bbfdf60aa4e
md"""
## Application 2 (Ex 5.8)
Consider the game illustrated below:


$(PlutoUI.LocalResource("./img/game2.png", :width => 300, :align => "middle"))
"""

# ╔═╡ b1e4afa4-895d-11eb-3f54-a32c30da203c
md"""
Player A moves first. The two players take turns moving, and each player must move his token to an open adjacent space in either direction. If the opponent occupies an adjacent space, then a player may jump over the opponent to the next open space if any. (For example, if A is on 3 and B is on 2, then A may move back to 1.) The game ends when one player reaches the opposite end of the board. If player A reaches space 4 first, then the value of the game to A is +1; if player B reaches space 1 first, then the value of the game to A is −1.

1. Draw the complete game tree, using the following conventions:
    * Write each state as $(s_A, s_B )$, where $s_A$ and $s_B$ denote the token locations.
    * Put each terminal state in a square box and write its game value in a circle.
    * Put loop states (states that already appear on the path to the root) in double square boxes. Since their value is unclear, annotate each with a “?” in a circle.
2. Now mark each node with its backed-up minimax value (also in a circle). Explain how you handle the “?” values and why. You should provide small reasoning.
3. Explain why the standard minimax algorithm would fail on this game tree and briefly sketch how you might fix it, drawing on your answer to (2). Does your modified algorithm give optimal decisions for all games with loops?
"""

# ╔═╡ 6a252a9e-895e-11eb-0410-29ab7456d6f9
md"""
## Application 3 - minimax algorithm
We will implement the minimax algorithm and apply it on a game of tic-tac-toe.Before going into the details of the algorithm implementation, we need to think about what we need and how we represent a game. From the course we know that we need the following:
* \\( S_0\\): The initial state, which specifies how the game is set up at the start.
* \\( \texttt{player}(s)\\): Defines which player has the move in a state.
* \\( \texttt{actions}(s)\\): Returns the set of legal moves in a state.
* \\( \texttt{result}(s,a)\\): The transition model, which defines the result of a move.
* \\( \texttt{terminal-test}(s)\\): A terminal test, which is true when the game is over and false otherwise.
* \\( \texttt{utility}(s, p)\\): A utility function (also called an objective function or payoff function), defines the final numeric value for a game that ends in terminal state s for a player p
* \\( \texttt{display}(g)\\): Display a game in way that is easy to interpret for us humans.

Below you find an implementation of tic-tac-toe. As with previous search methods, we provide a generic framework you can use to implement other games as well.
"""

# ╔═╡ 5c0a3b24-895f-11eb-33aa-8d74ee1eaacd
begin
	abstract type AbstractGame end 
	abstract type AbstractGameState end
	
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
end

# ╔═╡ 1e045506-8966-11eb-2c2f-f39e9d812b3b
md"""
### Illustration of a small game
"""

# ╔═╡ 298bb4d2-8966-11eb-2d9e-e15e24682397
begin
	# Initialise the game and show it's layout
	G = TicTacToeGame()
	state = G.initial
	display(G, state)
	# make random moves
	for _ = 1:9
		state = result(G, state, rand(actions(G, state)))
		display(G,state);println()
	end
end

# ╔═╡ 38c21f1c-8967-11eb-32ed-8bc5633cefd0
md"""
Note: for this illustration we do not check if a player wins during the (random) game,  we just fill the entire board to illustrate the game is working as intended.
"""

# ╔═╡ 4c136544-8967-11eb-322c-952f52cbf62f
@show state

# ╔═╡ 7711bc0c-8967-11eb-0fdf-e30e182b06b1
@assert terminal_test(G, state)

# ╔═╡ 8ae142e6-8967-11eb-32fb-7f214a71d8b5
md"""
### Minimax implementation
Now that we have evaluated that our game that is fully functional, we implement the minimax algorithm.
"""

# ╔═╡ b6ed262a-8967-11eb-3326-0398ad5931d5
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
end

# ╔═╡ abb41398-8969-11eb-29d0-05fe2438b3a8
md"""
### Illustration of the workings of minimax
Below we have a game that is near completion (only two cases left). Player $X$ has the turn. It is clear that the best possible move is to choose $(3,3)$, as this will lead to a victory for $X$. We see that the minimax algorithm correctly identifies this move. 
"""

# ╔═╡ c6a5b54e-8969-11eb-056e-9d4dd769f7c4
let
	G = TicTacToeGame()
	state = TicTacToeState("X",0, Dict((1,1)=>"X", (1,2)=>"O", (1,3)=>"X", (2,1)=>"O", (2,2)=>"X",(3,1)=>"X",(3,2)=>"O"), Tuple{Int64,Int64}[(2, 3), (3, 3)])
	display(G,state)
	move = minimax_decision(state, G);
	println("Player $(state.turn)'s best possible move: $(move)")
end

# ╔═╡ a65acd26-896c-11eb-21f0-3b8479d17e1c
md"""
### Actual game playing:
Try to realise the following:

* have two "intelligent" agents play against eachother
* have one "intelligent" agent play against a "dumb" agent i.e. an agent that makes random moves

What are the outcomes? Is this to be expected?
"""

# ╔═╡ 15b8bc4a-8b08-11eb-1991-95df70d5d7ee


# ╔═╡ beaa55c2-896c-11eb-30b8-7bfafa9b7441
begin
	"""
		playintelligentgame(;method::Function=minimax_decision)

	Play a TicTacToe game between two intelligent players.

	method: `minimax_decision` or `alphabeta_full_search`
	"""
	function playintelligentgame(;method::Function=minimax_decision)
		G = TicTacToeGame()
		state = G.initial
		for _ in 1:9
			move = method(state, G)
			state = result(G, state, move)
			if terminal_test(G,state)
				if state.utility == 0
					println("It's a draw!")
				else
					println("winner: $(state.turn == "X" ? "O" : "X")")
				end
				break
			end
		end
		display(G,state)
	end
	
	"""
		playdumbgame(;method::Function=minimax_decision)

	Play a TicTacToe game between a dumb (random moves) and an intelligent player.

	method: `minimax_decision` or `alphabeta_full_search`
	"""
	function playdumbgame(;method::Function=minimax_decision)
		G = TicTacToeGame()
		state = G.initial
		for _ in 1:9
			move = state.turn == "X" ? rand(actions(G, state)) : method(state, G)
			state = result(G, state, move)
			if terminal_test(G,state)
				if state.utility == 0
					println("It's a draw!")
				else
					println("winner: $(state.turn == "X" ? "O" : "X")")
				end
				break
			end
		end
		display(G,state)
	end
end

# ╔═╡ e6309ebc-896c-11eb-2a7e-2fdb4660393c
@time playintelligentgame()

# ╔═╡ 709fa20a-896d-11eb-3d35-0909bf0124c3
@time playdumbgame()

# ╔═╡ f859e3d4-8a23-11eb-2f36-65ccdeb9a782


# ╔═╡ ff0b3db8-8a23-11eb-261e-5b3d3a549a19
md"""
## Application 4 - α-β pruning
On the illustration below, you see the state of a two-player, zero-sum game with the values of the terminal states. Which branches won’t be explored when using α − β pruning? Suppose nodes are evaluated from the left to the right.

$(PlutoUI.LocalResource("./img/abtree.png", :width => 300, :align => "middle"))


"""

# ╔═╡ bb6e0a6a-896d-11eb-13da-adbd86f43b93
md"""
## Application 5 - $\alpha\text{-}\beta$ pruning
In the previous application you have used and studied the minimax algorithm. Below you find an implementation of the  $\alpha\text{-}\beta$ pruning algorithm (still exploring the game tree to the bottom). Use it in the same setting as before (i.e. TicTacToe) and compare its performance with the standard minimax algorithm.
"""

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
	@time minimax_decision(state, G)
	@time alphabeta_full_search(state,G)
end

# ╔═╡ Cell order:
# ╟─7a65ebb8-8958-11eb-3c1f-cf1daed9200e
# ╠═78b36c68-895b-11eb-3754-2f78ee415f6c
# ╟─88a95bfa-895b-11eb-2336-b3e894079c4f
# ╟─0a3f61fa-895c-11eb-3fae-f3cef63c52c4
# ╟─6783207e-895a-11eb-2d87-bf236e515f97
# ╟─615c38a0-895c-11eb-26bf-2f78b04eed50
# ╟─544bdd62-895c-11eb-14fa-f15a675494f8
# ╟─4dac7eea-895d-11eb-1ece-7bbfdf60aa4e
# ╟─b1e4afa4-895d-11eb-3f54-a32c30da203c
# ╟─6a252a9e-895e-11eb-0410-29ab7456d6f9
# ╠═5c0a3b24-895f-11eb-33aa-8d74ee1eaacd
# ╟─1e045506-8966-11eb-2c2f-f39e9d812b3b
# ╠═298bb4d2-8966-11eb-2d9e-e15e24682397
# ╟─38c21f1c-8967-11eb-32ed-8bc5633cefd0
# ╠═4c136544-8967-11eb-322c-952f52cbf62f
# ╠═7711bc0c-8967-11eb-0fdf-e30e182b06b1
# ╟─8ae142e6-8967-11eb-32fb-7f214a71d8b5
# ╠═b6ed262a-8967-11eb-3326-0398ad5931d5
# ╟─abb41398-8969-11eb-29d0-05fe2438b3a8
# ╠═c6a5b54e-8969-11eb-056e-9d4dd769f7c4
# ╟─a65acd26-896c-11eb-21f0-3b8479d17e1c
# ╠═15b8bc4a-8b08-11eb-1991-95df70d5d7ee
# ╠═beaa55c2-896c-11eb-30b8-7bfafa9b7441
# ╠═e6309ebc-896c-11eb-2a7e-2fdb4660393c
# ╠═709fa20a-896d-11eb-3d35-0909bf0124c3
# ╠═f859e3d4-8a23-11eb-2f36-65ccdeb9a782
# ╟─ff0b3db8-8a23-11eb-261e-5b3d3a549a19
# ╟─bb6e0a6a-896d-11eb-13da-adbd86f43b93
# ╠═ee38ca70-896d-11eb-1256-97a4b1b3f321
# ╠═cd81ae40-8a22-11eb-04a3-1bb77668a99b
