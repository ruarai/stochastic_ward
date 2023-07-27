
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

num_thresholds = 8
num_particles = 1000


model_priors = [Normal(0, 1), Normal(0, 1), Normal(-8, 2)]
model_perturbs = [Normal(0, 0.1), Normal(0, 0.1), Normal(0, 0.1)]

sigma = Array{Vector{Float64}, 2}(undef, num_thresholds, num_particles)
weights = zeros(num_thresholds, num_particles)
particle_outputs = Array{model_state}(undef, num_particles)



omega = 2.0
num_stochastic_samples = 100
num_candidates = round(Int, num_particles + num_particles * omega)
candidate_errors = zeros(num_thresholds, num_candidates)
candidate_sigmas = Array{Vector{Float64}}(undef, num_candidates)

candidate_occupancies = zeros(num_candidates, n_days, 2)

for i in 1:num_thresholds
    println("Threshold $i...")
    Threads.@threads for p in 1:num_candidates
        params = []

        if i == 1
            candidate_sigmas[p] = create_prior(model_priors)
        else
            particle_ix_sample = wsample(1:num_particles, weights[i - 1, :])

            candidate_sigmas[p] = perturb_parameters(sigma[i - 1, particle_ix_sample], model_perturbs)
        end

        best_candidate = model_process(1, candidate_sigmas[p], context, Random.MersenneTwister())
        candidate_errors[i, p] = get_error(best_candidate, true_occupancy)

        for j in 1:(num_stochastic_samples - 1)
            test_candidate = model_process(1, candidate_sigmas[p], context, Random.MersenneTwister())
            test_candidate_error = get_error(test_candidate, true_occupancy)

            if test_candidate_error < candidate_errors[i, p]
                candidate_errors[i, p] = test_candidate_error
                best_candidate = test_candidate
            end
        end

        if i == num_thresholds
            for d in 1:n_days
                t = (d - 1) * n_steps_per_day + 1
                candidate_occupancies[p, d, 1] = get_total_ward_occupancy(best_candidate, t, d)
                candidate_occupancies[p, d, 2] = get_ward_outbreak_occupancy(best_candidate, d)
            end
        end
    end

    error_perm = sortperm(candidate_errors[i, :])

    for (p_to, p_from) in pairs(error_perm[1:num_particles])
        sigma[i, p_to] = candidate_sigmas[p_from]


        if i == 1
            weights[1, p_to] = 1.0
        else
            numer = prior_prob(sigma[i, p_to], model_priors)
            denom = 0
            for p_j in 1:num_particles
                denom += weights[i - 1, p_j] * perturb_prob(sigma[i, p_to], sigma[i - 1, p_j], model_perturbs)
            end

            weights[i, p_to] = numer / denom
        end
    end
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
sim_occupancy = candidate_occupancies[sortperm(candidate_errors[end,:])[p_test],:,1]

plot(sim_occupancy)
plot!(true_occupancy[:,1])

sim_outbreak = candidate_occupancies[sortperm(candidate_errors[end,:])[p_test],:,2]
plot(sim_outbreak)
plot!(outbreak_occupancy)