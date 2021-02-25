# code snipper to download git repository entirely (instead of zip file from site)

using GitCommand
git() do git
    downloadfolder = joinpath(homedir(),"Desktop")
    cd(downloadfolder)
    run(`$git clone https://github.com/B4rtDC/DS425-PS.git`)
end