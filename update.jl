using Pkg
# change to course environment
Pkg.activate(dirname(@__FILE__))
# also change pwd for git functionality!
cd(dirname(@__FILE__))

import Git
# pull latest updates
const git = Git.git()

run(`$git config pull.rebase false`)
run(`$git pull`)

include("config.jl")