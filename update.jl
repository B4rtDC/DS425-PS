using Pkg, GitCommand 
# change to course environment
Pkg.activate(dirname(@__FILE__))

# pull latest updates
git() do git
    run(`$git pull`)
end