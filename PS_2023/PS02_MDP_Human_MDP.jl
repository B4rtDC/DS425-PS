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

    """
        simple_value_iteration(nmax=100)

    Simple value iteration algorithm. The algorithm stops after nmax iterations.
    We do not look at a convergence criterion.
    """
    function simple_value_iteration(nmax=100; γ=0.9)
        V = Dict(s => 0. for s in Human_MDP_States)
        # iterate
        for _ in 1:nmax
            for s in Human_MDP_States
                    V[s] = maximum([sum([p * (Human_MDP_Reward_model[(s, a, s_prime)] + γ * V[s_prime]) for (p, s_prime) in Human_MDP_Transition_model[(s, a)]]) for a in Human_MDP_Actions[s]])
            end
        end

        return V
    end

    """
        self_stopping_value_iteration(;γ=0.9, ϵ=1e-1)

    Self-stopping value iteration algorithm. The algorithm stops when the
    difference between two consecutive iterations is smaller than ϵ * (1 - γ) / γ.

    Returns both the utilities and the number of iterations.
    """
    function self_stopping_value_iteration(;γ=0.9, ϵ=1e-1)
        V_new = Dict(s => 0. for s in Human_MDP_States)
        N_it = 0
        while true
            N_it += 1
            V_old = copy(V_new)
            δ = 0.
            for s in Human_MDP_States
                V_new[s] = maximum([sum([p * (Human_MDP_Reward_model[(s, a, s_prime)] + γ * V_old[s_prime]) for (p, s_prime) in Human_MDP_Transition_model[(s, a)]]) for a in Human_MDP_Actions[s]])
                δ = max(δ, abs(V_new[s] - V_old[s]));
            end
            # stopping condition
            if δ < ϵ * (1 - γ) / γ
                return V_new, N_it
            end
        end
    end

    """
        simple_policy_extraction(V)

    Extract the policy from the computed utilities.
    """
    function simple_policy_extraction(V; γ=0.9)
        Π = Dict()
        # iterate states
        for s in Human_MDP_States
            a_ind = argmax([sum([p * (Human_MDP_Reward_model[(s, a, s_prime)] + γ * V[s_prime]) for (p, s_prime) in Human_MDP_Transition_model[(s, a)]]) for a in Human_MDP_Actions[s]])
            Π[s] = Human_MDP_Actions[s][a_ind]
        end
        return Π
    end

    """
        utility_over_time(n_max=30)

    Plot the utility of each state over time.
    """
    function utility_over_time(n_max=30)
        V = Dict(s => Float64[] for s in Human_MDP_States)
        for n in 1:n_max
            V_temp = simple_value_iteration(n)
            for s in Human_MDP_States
                push!(V[s], V_temp[s])
            end
        end

        return V
    end

    """
        plot_utility_over_time(V)

    Plot the utility of each state over time.
    """
    function plot_utility_over_time(V)
        p = plot(xlabel="Iteration", ylabel="Utility value", title="Value Iteration", legend=:topleft)
        for s in Human_MDP_States
            plot!(p, 1:length(V[s]), V[s], label=s)
        end

        savefig(p, "Human_MDP_value_iteration.pdf")
        return p
    end

    """
        plot_losses(V_evol)

    Plot the losses over time.
    """
    function plot_losses(V_evol, Π_opt; γ=0.9)
        ## utility error 
        errmax = Vector{Float64}()
        for i in 1:length(V_evol["Tired"])
            push!(errmax, maximum(abs(V_evol[s][end]-V_evol[s][i]) for s in Human_MDP_States))
        end
        p = plot(1:length(errmax), label="Max Error", errmax, xlabel="Iteration", ylabel="Maximum Error", title="Error over time", legend=:topright)

        # policy loss
        policy_loss = Vector{Float64}()
        for i in 1:length(V_evol["Tired"])
            # extract the policy from the current utility values => an action for each state
            π = simple_policy_extraction(Dict(s => V_evol[s][i] for s in Human_MDP_States))
            # compute the difference in utility between the current policy and the optimal policy 
            V_Π = [sum([p * (Human_MDP_Reward_model[(s, Π_opt[s], s_prime)] + γ * V_evol[s_prime][i]) for (p, s_prime) in Human_MDP_Transition_model[(s, Π_opt[s])]]) for s in Human_MDP_States]
            # compute the utilities for the current policy
            V_π = [sum([p * (Human_MDP_Reward_model[(s, π[s], s_prime)]     + γ * V_evol[s_prime][i]) for (p, s_prime) in Human_MDP_Transition_model[(s, π[s])]]) for s in Human_MDP_States]
            # compute the difference
            push!(policy_loss, maximum(abs.(V_π .- V_Π)))
        end
        
        plot!(p, 1:length(policy_loss), policy_loss, label="Policy Loss", xlabel="Iteration", legend=:topright)
        savefig(p, "Human_MDP_losses.pdf")
        return p
    end

    """
        plot_iterations_for_γ(npoints = 30)
    
    Plot the number of iterations to convergence for different values of γ.
    """
    function plot_iterations_for_γ(npoints = 30)
        res = Dict()
        p = plot(xlabel="γ", ylabel="Number of iterations", title="Impact of the discount factor", legend=:topleft, xlims=(0,1))
        for ϵ in [1e-1, 1e-2, 1e-3]
            res[ϵ] = Vector()
            for γ in range(0.1, stop=0.98, length=npoints)
                push!(res[ϵ], self_stopping_value_iteration(γ=γ, ϵ=ϵ)[2])
            end
            plot!(p, range(0.1, stop=0.98, length=npoints), res[ϵ], label="ϵ = $ϵ", yscale=:log10)
        end
        savefig(p, "Human_MDP_iterations.pdf")
        return p
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


# run value iteration
V_opt = Human_MDP.simple_value_iteration(1000)
# extract the policy
Π_opt = Human_MDP.simple_policy_extraction(V_opt) 
# make some plots
V_evol = Human_MDP.utility_over_time(75)
## utility over time 
p1 = Human_MDP.plot_utility_over_time(V_evol);
## error over time
p2 = Human_MDP.plot_losses(V_evol, Π_opt);
## impact of the discount factor
p3 = Human_MDP.plot_iterations_for_γ();

plot(p1, p2, p3, layout=(1,3), size=(1600, 1200))

