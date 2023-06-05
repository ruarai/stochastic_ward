
using Distributions
using StatsFuns
using DataFrames
using Random


include("globals.jl")
include("ward_epidemic.jl")


global def_n_ward_epidemic = 4
global ward_steady_state_size = 50




function run_sims(n_days, n_runs) 

    occ = zeros(n_days, n_runs)

    for i in 1:n_runs

        epi = ward_epidemic(
            fill(ward_steady_state_size, def_n_ward_epidemic),
            zeros(Int64, def_n_ward_epidemic),
            zeros(Int64, def_n_ward_epidemic),
        )

        log_ward_importation_rate = rand(Normal(-8, 1))
        log_ward_clearance_rate = log(1 / rand(TruncatedNormal(7, 4, 3, 14)))

        for t in 1:n_days
            epi = step_ward_epidemic(epi,
            rand(Poisson(100)),
            log_ward_importation_rate,
            log_ward_clearance_rate
            )

            occ[t, i] = sum(epi.Q)
        end
    end

    return occ

end

rng = MersenneTwister()

occ = @time run_sims(200, 50)

using Plots
plot(occ,
 legend = false, 
 ylim = (0, 100),
 size = (600, 400)
)
