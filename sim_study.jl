
include("inference_abc.jl")


function sample_from_prior(
    n_days,
    n_steps_per_day,

    case_curves,

    group_parameters_table,
    time_varying_estimates_table,

    parameters
)
    # Julia and R do not play along nicely with these parameters. Ensure they are rounded integers.
    n_days = round(Int32, n_days)
    n_steps_per_day = round(Int32, n_steps_per_day)

    n_steps = n_days * n_steps_per_day


    # Load in the bootstrapped data
    group_params = read_group_parameter_samples(group_parameters_table)
    time_varying_estimates = read_time_varying_estimates(time_varying_estimates_table, n_days)

    samples_cache = make_cached_samples(group_params[1], n_steps_per_day)


    # Produce a set of case curves that will be associated with a sample index (and not actually selected for)
    context_case_curves = Array{Array{Int32}}(undef, 1)
    context_case_curves[1] = case_curves[:, 1]

    context = model_context(
        n_steps_per_day, n_steps, n_days,
        time_varying_estimates[1], group_params[1],
        samples_cache, context_case_curves
    )

    state = model_process(1, parameters, context, Random.MersenneTwister())

    occupancy = zeros(n_days, 3)
    
    for d in 1:n_days
        i = (d - 1) * n_steps_per_day + 1
        occupancy[d, 1] = get_total_ward_occupancy(state, i, d)
        occupancy[d, 2] = get_total_ICU_occupancy(state, i)
        occupancy[d, 3] = get_ward_outbreak_occupancy(state, d)
    end

    return (context = context, occupancy = occupancy)
end

function infer_abc_decimation(context, true_occupancy)

    n_days = context.n_days
    n_steps_per_day = context.n_steps_per_day

    num_thresholds = 4
    num_particles = 1000


    model_priors = [Normal(0, 1), Normal(0, 1), Normal(-8, 2)]
    model_perturbs = [Normal(0, 0.1), Normal(0, 0.1), Normal(0, 0.1)]

    sigma = Array{Vector{Float64}, 2}(undef, num_thresholds, num_particles)
    weights = zeros(num_thresholds, num_particles)



    omega = 2.0
    num_stochastic_samples = 100
    num_candidates = round(Int, num_particles + num_particles * omega)
    candidate_errors = zeros(num_thresholds, num_candidates)
    candidate_sigmas = Array{Vector{Float64}}(undef, num_candidates)

    candidate_occupancies = zeros(num_candidates, n_days, 3)

    for i in 1:num_thresholds
        println("Threshold $i...")
        Threads.@threads for p in 1:num_candidates
            if i == 1
                candidate_sigmas[p] = create_prior(model_priors)
            else
                particle_ix_sample = wsample(1:num_particles, weights[i - 1, :])

                candidate_sigmas[p] = perturb_parameters(sigma[i - 1, particle_ix_sample], model_perturbs)
            end

            best_candidate = model_process(1, candidate_sigmas[p], context, Random.MersenneTwister())
            candidate_errors[i, p] = get_error(best_candidate, true_occupancy, n_days, n_steps_per_day)

            for j in 1:(num_stochastic_samples - 1)
                test_candidate = model_process(1, candidate_sigmas[p], context, Random.MersenneTwister())
                test_candidate_error = get_error(test_candidate, true_occupancy, n_days, n_steps_per_day)

                if test_candidate_error < candidate_errors[i, p]
                    candidate_errors[i, p] = test_candidate_error
                    best_candidate = test_candidate
                end
            end

            if i == num_thresholds
                for d in 1:n_days
                    t = (d - 1) * n_steps_per_day + 1
                    candidate_occupancies[p, d, 1] = get_total_ward_occupancy(best_candidate, t, d)
                    candidate_occupancies[p, d, 2] = get_total_ICU_occupancy(best_candidate, t)
                    candidate_occupancies[p, d, 3] = get_ward_outbreak_occupancy(best_candidate, d)
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


    sim_occupancy = candidate_occupancies[sortperm(candidate_errors[end,:])[1:num_particles],:,:]

    simulations_output = DataFrame(
        particle = Int[],
        day = Int[],

        sim_ward = Int[],
        sim_ward_outbreak = Int[], 
        sim_ICU = Int[]
    )


    parameters_output = DataFrame(
        particle = Int[],
        threshold = Int[],

        adj_pr_hosp = Float64[],
        adj_los = Float64[],

        log_importation_rate = Float64[],

        weight = Float64[]
    )

    for p in 1:num_particles
        for i in 1:num_thresholds
            push!(
                parameters_output,
                Dict(
                    :particle => p,
                    :threshold => i,
    
                    :adj_pr_hosp => sigma[i, p][1],
                    :adj_los => sigma[i, p][2],
                    :log_importation_rate => sigma[i, p][3],

                    :weight => weights[i, p],
                )
            )
        end

        for d in 1:n_days

            push!(
                simulations_output,
                Dict(
                    :particle => p,
                    :day => d,

                    :sim_ward => sim_occupancy[p, d, 1],
                    :sim_ICU => sim_occupancy[p, d, 2],
                    :sim_ward_outbreak => sim_occupancy[p, d, 3],
                )
            )
        end
    end

    return (
        simulations = simulations_output,
        parameters = parameters_output
    )
end




function get_error(state, true_occupancy, n_days, n_steps_per_day)
    sim_occupancy = zeros(n_days)

    for d in 1:n_days
        t = (d - 1) * n_steps_per_day + 1
        sim_occupancy[d] = get_total_ward_occupancy(state, t, d)
    end

    return sum(abs.(sim_occupancy .- true_occupancy[:,1]))
end