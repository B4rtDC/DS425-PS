# change to course environment
using Pkg
Pkg.activate(dirname(@__FILE__))

# start Pluto server
using Pluto
Pluto.run()