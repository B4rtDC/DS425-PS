# DS425

This page contains the support material for the practical sessions of the course DS425. 

Other related material for the course can be found on BelADL.


## Course overview
The course has multiple parts: reinforcement learning, machine learning & search methods. This year we start with machine learning because the project is related to this topic as well. 

### Evaluation:
- written project on machine learning learning, with a focus on leveraging large language models (LLMs). The concept of the projects is detailed in the 'Project_2024' folder.
- written test.

We will mainly be using Matlab (≥2022a) for the machine learning part (you will use this in future robotics courses as well) and Julia (≥1.8.1).

## Getting started
Get the repository:
- on Linux/Mac (in terminal): `git clone https://github.com/B4rtDC/DS425-PS.git`
- on Windows: run `downloadwithgit.jl`, where you modify `downloadfolder` with the proper path (*note*: this also works for Linux/Mac)
### Julia
Get Julia (also on your CDN laptop via the software center). We use a specific virtual environment for this course (located in this folder). If you follow the steps below, this environment will be activated automatically.
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

## Reading list:
* [Artificial Intelligence: A Modern Approach, 4th ed.](http://aima.cs.berkeley.edu)
* [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html)
* [The Hundred-Page Machine Learning Book](http://themlbook.com)
* [Introduction to Data Mining](https://www-users.cse.umn.edu/~kumar001/dmbook/index.php)
* [Principles of Data Mining](https://dl.acm.org/doi/10.5555/2462612)
* [Machine Learning: A Bayesian and Optimization Perspective 2nd ed.](https://www.elsevier.com/books/machine-learning/theodoridis/978-0-12-818803-3)

