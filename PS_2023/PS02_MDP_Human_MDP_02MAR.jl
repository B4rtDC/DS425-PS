using Pkg
Pkg.activate(joinpath(@__DIR__,".."))

module Human_MDP
    using Plots

    #  All possible states
    const Human_MDP_States = Set(["Energized"; "Tired"; "Healthy"])

    # Human_MDP_Actions: a dict of states mapping to possible actions
    const Human_MDP_Actions = Dict("Energized" => ["work"; "workout"],
                                      "Tired" => ["work"; "workout"; "sleep"],
                                      "Healthy" => ["work"])

    # Human_MDP_Transition model (s,a) -> {p(s'|s,a), s'}
    const Human_MDP_Transition_model = Dict(
        ("Energized", "work") => [(0.8, "Energized"); (0.2, "Tired")],
        ("Energized", "workout") => [(1.0, "Healthy")],
        ("Tired", "work") => [(1.0, "Tired")],
        ("Tired", "workout") => [(0.6, "Tired"); (0.4, "Energized")],
        ("Tired", "sleep") => [(6/7, "Energized"); (1/7, "Tired")],
        ("Healthy", "work") => [(1.0, "Tired")])

    # Human_MDP_Reward_model (s,a,s') -> r(s,a,s')
    const Human_MDP_Reward_model = Dict(
        ("Energized", "work", "Energized") => 5.0,
        ("Energized", "work", "Tired") => 1.0,
        ("Energized", "workout", "Healthy") => 4.0,
        ("Tired", "work", "Tired") => -3.0,
        ("Tired", "workout", "Tired") => 200.0,
        ("Tired", "workout", "Energized") => 300.0,
        ("Tired", "sleep", "Energized") => 6.0,
        ("Tired", "sleep", "Tired") => -1.0,
        ("Healthy", "work", "Tired") => 0.5)

    """
        possible_actions(s::String)

    Return the possible actions for a given state.
    """
    possible_actions(s::String) = Human_MDP_Actions[s]

    """
        apply_action(s::String, a::String)

    Apply an action to a state and return the new state. The new state is obtained
    by leveraging the transition model. We use the cumulative probability to pick a
    random outcome.
    """
    function apply_action(s::String, a::String)
        # get possible outcomes
        outcomes = Human_MDP_Transition_model[(s, a)]
        # pick a random outcome
        p_tot, ref_value = 0.0, rand()
        # find the outcome
        for (p, s_prime) in outcomes
            p_tot += p
            if p_tot >= ref_value
                return s_prime
            end
        end
    end

end


using StatsBase
# current state
mystate = "Energized"
# possible actions
my_possible_actions = Human_MDP.possible_actions(mystate)
# pick a random action
my_action = rand(my_possible_actions)
# possible outcomes
my_possible_outcomes = Human_MDP.Human_MDP_Transition_model[(mystate, my_action)]
# apply a specific action
my_action = "work"
my_new_state = Human_MDP.apply_action(mystate, my_action)
# sanity check for our transition model
countmap([Human_MDP.apply_action(mystate, my_action) for _ in 1:100]) # => 80% Energized, 20% Tired (as expected)


# What's next?
# - Simple Value Iteration
function value_iteration(;γ=0.9, Nmax=100)
    # initialize the values (utilities)
    V = Dict(s => 0.0 for s in Human_MDP.Human_MDP_States)
    # do the required number of iterations
    for _ in 1:Nmax
        # for each state compute the update
        for s in Human_MDP.Human_MDP_States
            # get the possible actions for state s
            actions = Human_MDP.possible_actions(s)
            # loop over the actions to find the one maximizing the utility value
            max_value = -Inf
            for a in actions
                # compute the update
                res = sum(T * (Human_MDP.Human_MDP_Reward_model[(s,a,s_prime)] + γ * V[s_prime]) for (T, s_prime) in Human_MDP.Human_MDP_Transition_model[(s,a)])
                # check if it is the largest value up to now
                if res > max_value
                    max_value = res
                end
            end
            # update the value
            V[s] = max_value
        end
    end
    return V
end

# Utility valyes
V_star = value_iteration(Nmax=1000)
# Extraction of the policy?
function policy_extraction(V; γ=0.9)
    Π = Dict(s => "?" for s in Human_MDP.Human_MDP_States)   
    for s in Human_MDP.Human_MDP_States
        # get the possible actions for state s
        actions = Human_MDP.possible_actions(s)
        # loop over the actions to find the one maximizing the utility value
        max_value, winner = -Inf, ""
        for a in actions
            # compute the update
            res = sum(T * (Human_MDP.Human_MDP_Reward_model[(s,a,s_prime)] + γ * V[s_prime]) for (T, s_prime) in Human_MDP.Human_MDP_Transition_model[(s,a)])
            # check if it is the largest value up to now
            if res > max_value
                max_value = res
                winner = a
            end
        end
        # update the value
        Π[s] = winner
    end
    return Π
end
policy_extraction(V_star)









begin
# How fast am I converging?
using Plots
# number of iterations
N = collect(range(0, 200, step=5))
H = Float64[]
T = Float64[]
E = Float64[]
for n in N
    # compute utilities given the number of iterations
    v = value_iteration(Nmax=n)
    # push the restults in the arrays
    push!(H, v["Healthy"])
    push!(T, v["Tired"])
    push!(E, v["Energized"])
end
# plot the results
begin
    convergence_plot = plot(N, H, label="Healthy", xlabel="Iteration", ylabel="utility")
    plot!(convergence_plot, N, T, label="Tired")
    plot!(convergence_plot,N, E, label="Energized")
end
end

# - Make an illustration

