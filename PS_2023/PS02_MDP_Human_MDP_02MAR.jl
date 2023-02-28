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
        ("Tired", "workout", "Tired") => -2.0,
        ("Tired", "workout", "Energized") => 3.0,
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
# - Extraction of the policy
# - Make an illustration


