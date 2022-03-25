# change to course environment
using Pkg
Pkg.activate(@__DIR__)

# start Pluto server
using Pluto
Pluto.run()