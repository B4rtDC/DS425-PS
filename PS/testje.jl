
"""
    turn_heading

Given an input heading, return the heading in inc positions further in a clockwise manner 

The headings can be specified with H. The heading h should occur in the collection of headings H.                 
"""
function turn_heading(h::Tuple{Int64,Int64}, inc::Int64; H=[(0,1),(1,0),(0,-1),(-1,0)])
    # get index of current heading
    i = findfirst(x->x==h, H)
    # return incremented heading
    return A[(i + inc + length(H) - 1) % length(H) + 1]
end

turn_heading((2,3),1)
