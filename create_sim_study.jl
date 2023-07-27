
using Plots
include("inference_abc.jl")


context = load_object("sim_study_context.jld2");


#model_priors = [Normal(0, 1), Normal(0, 1), Normal(-8, 1), Normal(-1, 1)]
#parameters = rand.(model_priors)
parameters = [1.0, 0.6, -5.5]

state = model_process(1, parameters, context, Random.MersenneTwister())

save_object("state.jld2", state)

#state = load_object("state.jld2")


n_days = context.n_days
n_steps_per_day = context.n_steps_per_day


true_occupancy = zeros(n_days, 2)
outbreak_occupancy = zeros(n_days)

for d in 1:n_days
    i = (d - 1) * n_steps_per_day + 1
    true_occupancy[d, 1] = get_total_ward_occupancy(state, i, d)
    true_occupancy[d, 2] = get_total_ICU_occupancy(state, i)
    outbreak_occupancy[d] = get_ward_outbreak_occupancy(state, d)
end

plot(true_occupancy)
plot!(outbreak_occupancy)

function get_error(state, true_occupancy)
    sim_occupancy = zeros(n_days)

    for d in 1:n_days
        t = (d - 1) * n_steps_per_day + 1
        sim_occupancy[d] = get_total_ward_occupancy(state, t, d)
    end

    return sum(abs.(sim_occupancy .- true_occupancy[:,1]))
end



histogram([sigma[end,j][3] for j in 1:num_particles])
plot!([parameters[3]], seriestype= :vline, label="", lc = "black")


plot(log.(weights'))
plot([sigma[i,j][1] for i in 1:num_thresholds, j in 1:num_particles], lc = "grey40", la = 0.1, legend = false)
plot!([1, num_thresholds], [state.adj_pr_hosp, state.adj_pr_hosp], lc = "red")

plot([sigma[i,j][2] for i in 1:num_thresholds, j in 1:num_particles], lc = "grey40", la = 0.1, legend = false)
plot!([1, num_thresholds], [state.adj_los, state.adj_los], lc = "black")

plot([sigma[i,j][3] for i in 1:num_thresholds, j in 1:num_particles], lc = "grey40", la = 0.1, legend = false)
plot!([1, num_thresholds], [state.log_ward_importation_rate, state.log_ward_importation_rate], lc = "black")


#plot(log.(candidate_errors), legend = false)

p_test = 1
sim_occupancy = candidate_occupancies[sortperm(candidate_errors[end,:])[1:num_particles],:,1]

plot(sim_occupancy')
plot!(true_occupancy[:,1])

sim_outbreak = candidate_occupancies[sortperm(candidate_errors[end,:])[p_test],:,2]
plot(sim_outbreak)
plot!(outbreak_occupancy)