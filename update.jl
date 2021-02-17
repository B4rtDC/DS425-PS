using Pkg
# change to course environment
Pkg.activate(dirname(@__FILE__))


using GitCommand
# pull latest updates
git() do git
    run(`$git pull`)
end