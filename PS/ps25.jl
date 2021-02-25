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

struct GNode 
    state::String
    parent::Union{GNode, Nothing}
    action::Union{Nothing}
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
    nodes = Vector{GNode}
    current = n
    while !isnothing(current.parent)
        push!(nodes, current)
        current = current.parent
    end
    push!(nodes, current.parent)
    return nodes
end

myproblem = GProblem("ABBC", "E")
root_node = GNode(myproblem.initialstate, nothing, nothing, 0)
@info myproblem, root_node
#@warn typeof(root_node)(myproblem.initialstate, nothing, nothing, 0)
@info Gfindactions(root_node)
@info Gapplyaction(2:3,root_node)