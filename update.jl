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

#=
git() do git
    run(`$git config pull.rebase false`)
    # if on CDN 
    #run(`$git config --global http.proxy http://CDNusername:CDNpassword@dmzproxy005.idcn.mil.intra:8080`)
    run(`$git pull`)
end
=#
include("config.jl")