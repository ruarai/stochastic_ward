
using Distributions
using StatsFuns
using DataFrames
using Random

using ParticleFilters

include("globals.jl")
include("group_parameters.jl")

include("ward_epidemic.jl")

include("progression.jl")
include("progression_pf.jl")

include("particle_filter/includes.jl")

include("pf_state.jl")
include("pf_context.jl")



function run_inference(
    n_days,
    n_steps_per_day,
    num_particles,

    case_curves,

    group_parameters_table,
    time_varying_estimates_table,

    true_occupancy_matrix
)

    n_days = round(Int32, n_days)
    n_steps_per_day = round(Int32, n_steps_per_day)

    n_steps = n_days * n_steps_per_day

    num_particles = round(Int64, num_particles)

    # Load in the bootstrapped data
    group_params = read_group_parameter_samples(group_parameters_table)
    time_varying_estimates = read_time_varying_estimates(time_varying_estimates_table, n_days)
    

    rng = MersenneTwister()

    # Initialise the components of the particle filter
    predict_model = ThreadedPredictModel{pf_state}(pf_step)
    reweight_model = ReweightModel(pf_prob_obs)
    resample_model = LowVarianceResamplerDebug(num_particles)
    filter = BasicParticleFilter2(predict_model, reweight_model, resample_model, num_particles, MersenneTwister())

    # Create the ParticleCollection, initialising values from the prior distribution
    particle_collection = ParticleCollection(
        [create_prior(case_curves, n_steps, n_steps_per_day) for i in 1:num_particles]
    );

    results_table = DataFrame(
        particle = Int[],
        day = Int[],

        sim_ward = Int[],
        sim_ward_outbreak = Int[], 
        sim_ICU = Int[],

        adj_pr_hosp = Float64[],
        adj_los = Float64[],
        cases = Float64[],

        importation_rate = Float64[],
        clearance_rate = Float64[],

        weight = Float64[]
    )

    weights_memory = zeros(num_particles)

    for d in 1:n_days

        t = (d - 1) * n_steps_per_day + 1

        ctx_day = pf_context(
            n_steps_per_day, n_steps, t,
            time_varying_estimates[1], group_params[1]
        )


        if d == n_days - 28
            println("Reinitialising case curves across particles")
            particle_collection = reinitialise_case_curves(particle_collection, num_particles, case_curves)
        end


        if true_occupancy_matrix[d, 1] < -0.5
            println("Stepping without inference at day $d")

            next_step = predict(filter, particle_collection, ctx_day, rng)
            particle_collection = ParticleCollection(next_step)
        else
            println("Stepping with inference at day $d")
            particle_collection = update(filter, particle_collection, ctx_day, true_occupancy_matrix[d, :])

            weights_memory = get_weights(reweight_model, particle_collection, ctx_day, true_occupancy_matrix[d, :])
        end


        for p in 1:num_particles
            particle_p = particle(particle_collection, p)

            push!(
                results_table,
                Dict(
                    :particle => p,
                    :day => d,

                    :sim_ward => get_total_ward_occupancy(particle_p, t),
                    :sim_ward_outbreak => get_outbreak_occupancy(particle_p, t),
                    :sim_ICU => get_total_ICU_occupancy(particle_p, t),
                    :adj_pr_hosp => particle_p.adj_pr_hosp,
                    :adj_los => particle_p.adj_los,
                    :clearance_rate => particle_p.epidemic.clearance_rate,
                    :importation_rate => particle_p.epidemic.importation_rate,
                    :cases => particle_p.case_curve[d],
                    :weight => weights_memory[p]
                )
            )
        end
    end



    return results_table
end

function create_prior(
    case_curves, n_steps, n_steps_per_day
)
    case_curve = case_curves[:, 1]

    adj_pr_hosp = rand(Normal(0, 0.4))
    adj_los = rand(Normal(0, 0.4))

    epidemic_importation_rate = rand(LogNormal(-8, 1))
    clearance_rate = 1 / rand(TruncatedNormal(7, 4, 3, 14))

    return pf_state(
        zeros(def_n_age_groups, n_steps, def_n_compartments, def_n_slots),

        adj_pr_hosp, adj_los,

        case_curve,

        ward_epidemic(
            fill(ward_steady_state_size, def_n_ward_epidemic), zeros(Int64, def_n_ward_epidemic), zeros(Int64, def_n_ward_epidemic),
            epidemic_importation_rate, clearance_rate
        )
    )
end

function reinitialise_case_curves(particles, num_particles, case_curves)
    particle_vec = Vector{pf_state}(undef, num_particles)

    for i in 1:num_particles
        pf_state_old = particle(particles, i)

        particle_vec[i] = pf_state(
            pf_state_old.arr_all,

            pf_state_old.adj_pr_hosp,
            pf_state_old.adj_los,

            case_curves[:, sample(1:size(case_curves, 2))],

            ward_epidemic(
                pf_state_old.epidemic.S, pf_state_old.epidemic.I, pf_state_old.epidemic.Q,
                pf_state_old.epidemic.importation_rate, pf_state_old.epidemic.clearance_rate
            )
        )
    end

    return ParticleCollection(particle_vec)
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