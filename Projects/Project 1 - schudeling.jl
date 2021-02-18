### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 0e4dcd02-71f8-11eb-2ac3-612951cdb3d4
md"""
# Examen scheduling
The purpose of the first project is to create an exam schedule. This is a group work where the groups are in competition with each other. The evaluation is made on the basis of:
* the correctness of the proposed solution
* the computational cost
* the functionality
* the scalability
* the generalizability

If the results meet expectations, your work can also be used effectively in the future for planning exams or class schedules.

The absolute minimum requirements (or constraints) that must be respected are
* the availability of professors (and teaching assistants)
* the non-overlapping of subjects

possible additions are:
* distinction between an oral or written exam
* whether or not it is compulsory to conjoin several examination days for the same course if it runs over several days
* courses that are evaluated in several parts (possibly by several professors)
* preferences (e.g. certain minimum number of days of preparation for a certain course, taking into account student's input)

You may consider this a constraint satisfaction problem. Different way of solving it can and may be used (backtracking <> local search etc.). 

Start with one year, then extend the functionality to one faculty and finally to both faculties.
"""

# ╔═╡ Cell order:
# ╟─0e4dcd02-71f8-11eb-2ac3-612951cdb3d4
