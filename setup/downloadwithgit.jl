# code snipper to download git repository entirely (instead of zip file from site)
import Pkg
Pkg.add("Git")

import Git
const git = Git.git()

downloadfolder = joinpath(homedir(),"Desktop")
cd(downloadfolder)

run(`$git clone https://github.com/B4rtDC/DS425-PS.git`)
