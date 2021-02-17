using Pkg
# change to course environment
Pkg.activate(dirname(@__FILE__))

using GitCommand
# pull latest updates
git() do git
    run(`$git config pull.rebase false`)
    run(`$git pull`)
end

include("config.jl")