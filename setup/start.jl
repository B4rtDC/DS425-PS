# change to course environment
using Pkg
Pkg.activate(joinpath(@__DIR__,".."))

# start Pluto server
using Pluto
Pluto.run()