# DS425

This page contains the support material for the practical sessions of the course DS425. 

Other related material for the course can be found on BelADL.

We will use Julia 1.5.3 (also on your CDN laptop) in combination with Pluto notebooks. We use a specific virtual environment for this course (located in this folder). If you follow the steps below, 
this environment will be activated automatically.

## Getting started
1. Run the config file to check you are up-to-date with the dependencies by running the `config.jl`. 
2. Start the Pluto server by running `start.jl`.

## Getting updates
To sync with the latest version on GitHub, you can run `update.jl`. This should work on every computer, the only condition is that you are not using the CDN network.
 
In a terminal (in the folder):
```
julia config.jl
julia start.jl
julia update.jl
```
or within Julia:
```julia
include("config.jl")
include("start.jl")
include("update.jl")
```


