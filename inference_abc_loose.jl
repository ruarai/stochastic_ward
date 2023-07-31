

function run_inference_loose(
    n_days,
    n_steps_per_day,

    num_particles,
    num_threshold_steps,
    omega,
    num_stochastic_samples,

    case_curves,

    group_parameters_table,
    time_varying_estimates_table,

    true_occupancy_matrix
)
    # Julia and R do not play along nicely with these parameters. Ensure they are rounded integers.
    n_days = round(Int, n_days)
    n_steps_per_day = round(Int, n_steps_per_day)

    num_particles = round(Int, num_particles)
    num_threshold_steps = round(Int, num_threshold_steps)
    num_stochastic_samples = round(Int, num_stochastic_samples)

    n_steps = n_days * n_steps_per_day
    num_candidates = round(Int, num_particles + num_particles * omega)

    # Load in the bootstrapped data
    group_params = read_group_parameter_samples(group_parameters_table)
    time_varying_estimates = read_time_varying_estimates(time_varying_estimates_table, n_days)

    samples_cache = make_cached_samples(group_params[1], n_steps_per_day)

    # Produce a set of case curves that will be associated with a sample index (and not actually selected for)
    context_case_curves = make_context_case_curves(case_curves, num_candidates)

    context = model_context(
        n_steps_per_day, n_steps, n_days,
        time_varying_estimates[1], group_params[1],
        samples_cache, context_case_curves
    )

    # Define model priors and perturbation kernels
    model_priors = [Normal(0, 1), Normal(0, 1), Normal(-8, 2)]
    model_perturbs = [Normal(0, 0.1), Normal(0, 0.1), Normal(0, 0.1)]

    # Arrays to hold particle parameters (sigma) and weights
    sigma = Array{Vector{Float64}, 2}(undef, num_threshold_steps, num_particles)
    weights = zeros(num_threshold_steps, num_particles)

    # Arrays to hold candidate parameter values for each threshold step
    candidate_errors = zeros(num_threshold_steps, num_candidates)
    candidate_sigmas = Array{Vector{Float64}}(undef, num_candidates)

    # Hold best candidate occupancies for final threshold step
    candidate_occupancies = zeros(num_candidates, n_days, 3)

    for i in 1:num_threshold_steps
        println("Running threshold $i...")
        Threads.@threads for p in 1:num_candidates
            # Get parameter values from prior (first step) or from particle cloud (after first step)
            if i == 1
                candidate_sigmas[p] = create_prior(model_priors)
            else
                particle_ix_sample = wsample(1:num_particles, weights[i - 1, :])

                candidate_sigmas[p] = perturb_parameters(sigma[i - 1, particle_ix_sample], model_perturbs)
            end

            # Fixed maximum value for log_importation_rate
            candidate_sigmas[p][3] = min(candidate_sigmas[p][3], -5.0)

            # Run num_stochastic_samples runs with given parameters,
            # selecting the parameter set with the lowest error
            best_candidate = model_process(p, candidate_sigmas[p], context, Random.MersenneTwister())
            candidate_errors[i, p] = get_error(best_candidate, true_occupancy_matrix, n_days, n_steps_per_day)

            for j in 1:(num_stochastic_samples - 1)
                test_candidate = model_process(p, candidate_sigmas[p], context, Random.MersenneTwister())
                test_candidate_error = get_error(test_candidate, true_occupancy_matrix, n_days, n_steps_per_day)

                if test_candidate_error < candidate_errors[i, p]
                    candidate_errors[i, p] = test_candidate_error
                    best_candidate = test_candidate
                end
            end

            # For the final step, save occupancies from the best candidate run
            if i == num_threshold_steps
                for d in 1:n_days
                    t = (d - 1) * n_steps_per_day + 1
                    candidate_occupancies[p, d, 1] = get_total_ward_occupancy(best_candidate, t, d)
                    candidate_occupancies[p, d, 2] = get_total_ICU_occupancy(best_candidate, t)
                    candidate_occupancies[p, d, 3] = get_ward_outbreak_occupancy(best_candidate, d)
                end
            end
        end

        # Sort candidates by their error value,
        # then select the best n = num_particles performing candidates
        error_perm = sortperm(candidate_errors[i, :])
        for (ix_to, ix_from) in pairs(error_perm[1:num_particles])
            sigma[i, ix_to] = candidate_sigmas[ix_from]

            if i == 1
                weights[1, ix_to] = 1.0
            else
                numer = prior_prob(sigma[i, ix_to], model_priors)
                denom = 0
                for ix_j in 1:num_particles
                    denom += weights[i - 1, ix_j] * perturb_prob(sigma[i, ix_to], sigma[i - 1, ix_j], model_perturbs)
                end

                weights[i, ix_to] = numer / denom
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
        for i in 1:num_threshold_steps
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


function log_1p(x)
    return log(x + 1)
end

function get_error(state, true_occupancy, n_days, n_steps_per_day)
    error = 0

    for d in 1:n_days
        t = (d - 1) * n_steps_per_day + 1
        sim_ward_occupancy = get_total_ward_occupancy(state, t, d)
        sim_ICU_occupancy = get_total_ICU_occupancy(state, t)

        if true_occupancy[d, 1] > -0.5
            error += abs.(log_1p(sim_ward_occupancy) - log_1p(true_occupancy[d, 1]))
        end

        if true_occupancy[d, 2] > -0.5
            error += abs.(log_1p(sim_ICU_occupancy) - log_1p(true_occupancy[d, 2]))
        end

    end

    return error
end


