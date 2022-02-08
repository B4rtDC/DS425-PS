# DS425

This page contains the support material for the practical sessions of the course DS425. 

Other related material for the course can be found on BelADL.


## Course overview
The course has two big parts: Machine learning & search methods. This year we start with machine learning because the project is related to this topic as well. 

### Evaluation:
- written project on machine learning. The projects will be available in March. Once started, the hour on Fridays will be reserved for you to work on this. The projects will be done in groups of two with mutual competition.
- written test on search methods.

We will be using Matlab (≥2020b) for the machine learning part (you will use this in future robotics courses as well) and Julia (≥1.7.1) second part of the course.

## Getting started
Get the repository:
- on Linux/Mac (in terminal): `git clone https://github.com/B4rtDC/DS425-PS.git`
- on Windows: run `downloadwithgit.jl`, where you modify `downloadfolder` with the proper path (*note*: this also works for Linux/Mac)
### Julia
Get Julia 1.7.1 (also on your CDN laptop via the software center). We use a specific virtual environment for this course (located in this folder). If you follow the steps below, this environment will be activated automatically.
1. Run the config file to check you are up-to-date with the dependencies by running the `config.jl`. 
2. Start the Pluto server by running `start.jl`.

### Matlab
Get Matlab on your machine (also on your CDN laptop vie the software center). The following products are useful:

- Computer Vision Toolbox
- Deep Learning Toolbox (required)
- Optimization Toolbox
- Parallel Computing Toolbox
- Reinforcement Learning 
- Signal Processing Toolbox 
- Statistics and Machine Learning Toolbox (required)
- Text Analytics Toolbox (required)

## Getting updates
To sync with the latest version on GitHub, you can run `update.jl`. This should work on every computer. When connected to CDN by VPN, it should work. If you are working on the CDN machine, you need to pass your credentials over the proxy. This can be done by modifying and uncommenting line 10 in `update.jl`.

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

Note: this still requires you to have a .git record in your directory, otherwise it will not work.
