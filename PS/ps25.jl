# uninformed search illustration
using DataStructures # (for the queue)

const comb = Dict(  "AC"=>"E",
                    "AB"=>"BC",
                    "BB"=>"E",
                    "EA"=>"A",
                    "EB"=>"B",
                    "EC"=>"C",
                    "EE"=>"E")
const candidates = collect(keys(comb))

# generic types
abstract type Problem end;
abstract type Node end;

# specific (problem related) types
struct GProblem <: Problem
    initialstate::String
    goalstate::String
end
# update!!!
struct GNode <: Node
    state::String
    parent::Union{GNode, Nothing}
    action::Union{Nothing, UnitRange}
    pathcost::Int
end

Base.show(io::IO, n::GNode) = print(io, "$(n.state) on level $(n.pathcost)")
# functor: construct new instance of type GNode
function (::GNode)(state::String, parent::Union{GNode, Nothing}, 
                    action::Union{Nothing}, pathcost::Int)
    GNode(state, parent, action, pathcost)
end

function Gfindactions(n::GNode)
    return vcat([findall(cad, n.state) for cad in candidates]...)
end
function Gapplyaction(a::UnitRange, n::GNode)
    return n.state[1:a.start-1] * comb[n.state[a]] * n.state[a.stop+1:end]
end
function Ggoaltest(n::GNode, p::GProblem)
    return isequal(n.state, p.goalstate)
end
function Gsolution(n::GNode)
    nodes = Vector{GNode}()
    current = n
    while !isnothing(current.parent)
        push!(nodes, current)
        current = current.parent
    end
    push!(nodes, current)
    for i in length(nodes):-1:1
        @info nodes[i].state, nodes[i].action ,nodes[i].pathcost
    end
    return nodes
end

function treesearch(p::T, r::N; goaltest::Function, 
                                findactions::Function, 
                                applyaction::Function, 
                                solution::Function) where {T<:Problem, N<:Node}
    fringe = DataStructures.Queue{N}()       # create FIFO Queue
    enqueue!(fringe, r)                      # place root node in the Queue
    while !isempty(fringe)
        node = dequeue!(fringe)
        for action in findactions(node)
            child = N(applyaction(action, node) , node, action, node.pathcost + 1)
            if child ∉ fringe
                if goaltest(child, p)
                    return solution(child)
                end
                enqueue!(fringe, child)
            end
        end
    end
    @warn "failed to find a solution"
end;


myproblem = GProblem("ABBC", "E")
root_node = GNode(myproblem.initialstate, nothing, nothing, 0)
@info treesearch(myproblem, root_node, goaltest=Ggoaltest, 
            findactions=Gfindactions, applyaction=Gapplyaction, solution=Gsolution)