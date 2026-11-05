# Runs run.jl with its whole output written to full-log.txt next to it (the box returns only the log's tail).
out = joinpath(@__DIR__, "full-log.txt")
open(out, "w") do io
    redirect_stdio(; stdout = io, stderr = io) do
        try
            include(joinpath(@__DIR__, "run.jl"))
        catch err
            println(io, "RUN ERRORED: ", sprint(showerror, err))
        end
    end
end
for l in eachline(out)
    occursin(r"Error During Test|Test Failed|RUN ERRORED|Pass +Fail|Pass +Error", l) && println(l)
end
