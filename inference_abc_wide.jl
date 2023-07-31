


# The primary function for calling from R
function run_inference(
    n_days,
    n_steps_per_day,

    case_curves,

    group_parameters_table,
    time_varying_estimates_table,

    true_occupancy_matrix
)
    # Julia and R do not play along nicely with these parameters. Ensure they are rounded integers.
    n_days = round(Int32, n_days)
    n_steps_per_day = round(Int32, n_steps_per_day)

    n_steps = n_days * n_steps_per_day

    num_particles = 1000
    num_thresholds = 4


    omega = 3.0
    num_stochastic_samples = 100
    num_candidates = round(Int, num_particles + num_particles * omega)

    # Load in the bootstrapped data
    group_params = read_group_parameter_samples(group_parameters_table)
    time_varying_estimates = read_time_varying_estimates(time_varying_estimates_table, n_days)

    samples_cache = make_cached_samples(group_params[1], n_steps_per_day)

    # Produce a set of case curves that will be associated with a sample index (and not actually selected for)
    context_case_curves = Array{Array{Int32}}(undef, num_candidates)
    for i in 1:num_candidates
        context_case_curves[i] = case_curves[:, sample(1:size(case_curves, 2))]
    end

    context = model_context(
        n_steps_per_day, n_steps, n_days,
        time_varying_estimates[1], group_params[1],
        samples_cache, context_case_curves
    )


    model_priors = [Normal(0, 1), Normal(0, 1), Normal(-8, 2)]
    model_perturbs = [Normal(0, 0.1), Normal(0, 0.1), Normal(0, 0.1)]

    sigma = Array{Vector{Float64}, 2}(undef, num_thresholds, num_particles)
    weights = zeros(num_thresholds, num_particles)



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

            candidate_sigmas[p][3] = min(candidate_sigmas[p][3], -5.0)

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



function read_group_parameter_samples(group_parameters_table)

    n_samples = round(Int32, maximum(group_parameters_table.sample))
    println("Reading in $n_samples group_parameters across $def_n_age_groups age groups")

    out_samples = Vector{Vector{group_parameters}}(undef, n_samples)

    for i in 1:n_samples
        out_samples[i] = Vector{group_parameters}(undef, def_n_age_groups)
    end

    for row in eachrow(group_parameters_table)
        i_sample = round(Int32, row.sample)
        i_age_group = findfirst(x -> x == row.age_group, age_groups)

        out_samples[i_sample][i_age_group] = group_parameters(
            row.pr_ward_to_discharge,
            row.pr_ward_to_ICU,

            row.pr_ICU_to_discharge,
            row.pr_ICU_to_postICU,

            row.pr_postICU_to_death,
        
            Gamma(row.shape_onset_to_ward, row.scale_onset_to_ward),
            
            Gamma(row.shape_ward_to_discharge, row.scale_ward_to_discharge),
            Gamma(row.shape_ward_to_death, row.scale_ward_to_death),
            Gamma(row.shape_ward_to_ICU, row.scale_ward_to_ICU),
            
            Gamma(row.shape_ICU_to_death, row.scale_ICU_to_death),
            Gamma(row.shape_ICU_to_discharge, row.scale_ICU_to_discharge),
            Gamma(row.shape_ICU_to_postICU, row.scale_ICU_to_postICU),

            Gamma(row.shape_postICU_to_death, row.scale_postICU_to_death),
            Gamma(row.shape_postICU_to_discharge, row.scale_postICU_to_discharge)
        )
    end

    return out_samples
end


function read_time_varying_estimates(time_varying_estimates_table, n_days)

    n_days = round(Int32, n_days)

    n_samples = round(Int32, maximum(time_varying_estimates_table.bootstrap))
    println("Reading in $n_samples time-varying estimate samples across $def_n_age_groups age groups")

    out_samples = Vector{}(undef, n_samples)

    for i in 1:n_samples
        out_samples[i] = (
            pr_age_given_case = zeros(def_n_age_groups, n_days),
            pr_hosp = zeros(def_n_age_groups, n_days),
            pr_ICU = zeros(def_n_age_groups, n_days)
        )
    end


    for row in eachrow(time_varying_estimates_table)
        i_sample = round(Int32, row.bootstrap)
        i_age_group = findfirst(x -> x == row.age_group, age_groups)
        i_t = round(Int32, row.t)

        out_samples[i_sample].pr_age_given_case[i_age_group, i_t] = row.pr_age_given_case
        out_samples[i_sample].pr_hosp[i_age_group, i_t] = row.pr_hosp
        out_samples[i_sample].pr_ICU[i_age_group, i_t] = row.pr_ICU
    end



    return out_samples
    
end

# Returns the 'forecast start day'
# Defined as the latest day with occupancy data
function get_forecast_start_day(occ_data)
    for i in reverse(2:length(occ_data))
        if occ_data[i] < -0.5 && occ_data[i - 1] > -0.5
            return i - 1
        end
    end
    return -1
end