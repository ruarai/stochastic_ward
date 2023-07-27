
using Distributions
using StatsFuns
using DataFrames
using Random
using JLD2

include("globals.jl")
include("group_parameters.jl")

include("ward_epidemic.jl")

include("progression.jl")

include("model_step.jl")
include("observation.jl")

include("group_delay_samples_cache.jl")

include("model_state.jl")
include("model_context.jl")

include("model_parameters.jl")


# The primary function for calling from R
function run_inference(
    n_days,
    n_steps_per_day,
    num_particles,
    thresholds,
    rejections_per_selections,

    case_curves,

    group_parameters_table,
    time_varying_estimates_table,


    true_occupancy_matrix
)
    # Julia and R do not play along nicely with these parameters. Ensure they are rounded integers.
    n_days = round(Int32, n_days)
    n_steps_per_day = round(Int32, n_steps_per_day)
    num_particles = round(Int64, num_particles)
    num_thresholds = length(thresholds)

    n_steps = n_days * n_steps_per_day


    # Load in the bootstrapped data
    group_params = read_group_parameter_samples(group_parameters_table)
    time_varying_estimates = read_time_varying_estimates(time_varying_estimates_table, n_days)
    
    forecast_start_day = get_forecast_start_day(true_occupancy_matrix[:, 1])

    samples_cache = make_cached_samples(group_params[1], n_steps_per_day)

    rng = MersenneTwister()


    # Produce a set of case curves that will be associated with a sample index (and not actually selected for)
    context_case_curves = Array{Array{Int32}}(undef, num_particles)
    for i in 1:num_particles
        context_case_curves[i] = case_curves[:, sample(1:size(case_curves, 2))]
    end

    context = model_context(
        n_steps_per_day, n_steps, n_days,
        time_varying_estimates[1], group_params[1],
        samples_cache, context_case_curves
    )

    save_object("sim_study_context.jld2", context)
    return false
    


    model_priors = [Normal(0, 1), Normal(0, 1), Normal(-8, 1), Normal(-1, 1)]
    model_perturbs = [Normal(0, 0.01), Normal(0, 0.01), Normal(0, 0.01), Normal(0, 0.01)]

    sigmas = Array{Vector{Float64}, 2}(undef, num_thresholds, num_particles)
    weights = zeros(num_thresholds, num_particles)
    particle_outputs = Array{model_state}(undef, num_particles)

    for threshold_i in eachindex(thresholds)
        println("Threshold $threshold_i, ", thresholds[threshold_i])
        for particle_i in 1:num_particles
            rejected = true

            while rejected
                rejected = false

                if threshold_i == 1
                    sigmas[1, particle_i] = create_prior(model_priors)
                else
                    particle_ix_sample = wsample(1:num_particles, weights[threshold_i - 1, :])

                    sigmas[threshold_i, particle_i] = perturb_parameters(sigmas[threshold_i - 1, particle_ix_sample], model_perturbs)
                end

                particle_outputs[particle_i] = create_model_state(sigmas[threshold_i, particle_i], n_steps, n_days)

                for d in 1:n_days
                    t = (d - 1) * n_steps_per_day + 1
                    is_forecast = d >= forecast_start_day

                    particle_outputs[particle_i] = model_step(particle_outputs[particle_i], context, particle_i, Random.MersenneTwister())


                    if true_occupancy_matrix[d, 1] > -0.5

                        sim_ward = get_total_ward_occupancy(particle_outputs[particle_i], t, d)
                        sim_ICU = get_total_ICU_occupancy(particle_outputs[particle_i], t)

                        known_ward = true_occupancy_matrix[d, 1]
                        known_ICU = true_occupancy_matrix[d, 2]

                        error_ward = abs(known_ward - sim_ward)
                        error_ICU = abs(known_ICU - sim_ICU)

                        if error_ward > max(known_ward * thresholds[threshold_i], 2) || error_ICU > max(known_ICU * thresholds[threshold_i] * 1.5, 4)
                            rejected = true
                            break
                        end
                    end
                end
            end

            if threshold_i == 1
                weights[1, particle_i] = 1.0
            else
                numer = prior_prob(sigmas[threshold_i, particle_i], model_priors)
                denom = 0
                for particle_j in 1:num_particles
                    denom += weights[threshold_i - 1, particle_j] * perturb_prob(sigmas[threshold_i, particle_i], sigmas[threshold_i - 1, particle_j], model_perturbs)
                end

                weights[threshold_i, particle_i] = numer / denom
            end
        end
    end



    parameters_output = DataFrame(
        particle = Int[],

        adj_pr_hosp = Float64[],
        adj_los = Float64[],

        log_importation_rate = Float64[],
        log_clearance_rate = Float64[],
    )

    simulations_output = DataFrame(
        particle = Int[],
        day = Int[],

        sim_ward = Int[],
        sim_ward_outbreak = Int[], 
        sim_ward_progression = Int[],
        sim_ICU = Int[]
    )

    for p in 1:num_particles
        particle_p = particle_outputs[p]


        push!(
            parameters_output,
            Dict(
                :particle => p,

                :adj_pr_hosp => particle_p.adj_pr_hosp,
                :adj_los => particle_p.adj_los,
                :log_importation_rate => particle_p.log_ward_importation_rate,
                :log_clearance_rate => particle_p.log_ward_clearance_rate,
            )
        )

        for d in 1:n_days
            t = (d - 1) * n_steps_per_day + 1

            push!(
                simulations_output,
                Dict(
                    :particle => p,
                    :day => d,

                    :sim_ward => get_total_ward_occupancy(particle_p, t, d),
                    :sim_ward_outbreak => get_ward_outbreak_occupancy(particle_p, d),
                    :sim_ward_progression => get_ward_progression_occupancy(particle_p, t),
                    :sim_ICU => get_total_ICU_occupancy(particle_p, t),
                )
            )
        end
    end

    return (simulations = simulations_output, parameters = parameters_output)
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