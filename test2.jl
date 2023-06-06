include("inference_pf.jl")


using DelimitedFiles
using CSV

n_steps_per_day = 4
num_particles = 100

case_curves = readdlm("case_curves.csv", ',') 


group_parameters_table = DataFrame(CSV.File("param_tbl.csv"))
time_varying_estimates_table = DataFrame(CSV.File("morb_tbl.csv"))

true_occupancy_matrix = readdlm("occ_mat.csv", ',')

n_days = size(case_curves, 1)

x = @report_opt run_inference(
    n_days,
    n_steps_per_day,
    num_particles,

    case_curves,

    group_parameters_table,
    time_varying_estimates_table,

    true_occupancy_matrix
)




group_params = read_group_parameter_samples(group_parameters_table)



group_param_sample = group_params[1]

cache = make_cached_samples(group_param_sample, n_steps_per_day)


samples = make_delay_samples(group_param_sample, 1024, 1, 0.0)

x_los_adj = collect(-3:0.05:3)
samples_los_adj = Vector{group_delay_samples}(undef, length(x_los_adj))

for i in eachindex(x_los_adj)
    samples_los_adj[i] = make_delay_samples(group_param_sample, 1024, 4, x_los_adj[i])
end


findnearest(x_los_adj, 5)

cache = group_delay_samples_cache(x_los_adj, samples_los_adj)


get_cached_samples(cache, 3)