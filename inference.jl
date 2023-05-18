
using Distributions
using StatsFuns
using DataFrames
using Random


include("globals.jl")
include("group_parameters.jl")
include("progression.jl")


function run_inference(
    n_days,
    n_steps_per_day,
    n_particles,

    case_curves,

    group_parameters_table,
    time_varying_estimates_table,

    true_ward_vec,

    thresholds
)

    n_days = round(Int32, n_days)
    n_steps_per_day = round(Int32, n_steps_per_day)
    n_particles = round(Int32, n_particles)

    n_thresholds = length(thresholds)

    group_params = read_group_parameter_samples(group_parameters_table)
    time_varying_estimates = read_time_varying_estimates(time_varying_estimates_table, n_days)
    
    ward_vec_days = true_ward_vec .> -0.5


    weights = zeros(n_particles, n_thresholds)
    attempts = zeros(n_particles, n_thresholds)
    params = zeros(n_particles, n_thresholds)


    samples_out = zeros(n_particles, n_days)

    prior_pr_hosp = Normal(0, 1)
    perturb_kernel = Normal(0, 0.025)

    for t in 1:n_thresholds
        threshold = thresholds[t]
        println("t = $t ($threshold)")

        Threads.@threads for p in 1:n_particles
            accepted = false

            group_param_sample = sample(group_params)
            time_varying_sample = sample(time_varying_estimates)

            case_curve = case_curves[:, sample(1:size(case_curves, 2))]

            n_attempts = 0

            while !accepted
                if t == 1
                    params[p, t] = rand(prior_pr_hosp)
                else
                    params[p, t] = wsample(params[:, t - 1], weights[:, t - 1]) +
                        rand(perturb_kernel)
                end

                for j in 1:def_n_age_groups

                    pr_age_and_hosp = logistic.(
                        params[p, t] .+
                        logit.(
                            time_varying_sample.pr_age_given_case[j,:] .* 
                            time_varying_sample.pr_hosp[j,:]
                        )
                    )
                    
    
                    hosp_curve = zeros(n_days)
    
                    for d in 1:n_days
                        hosp_curve[d] = rand(Binomial(case_curve[d], pr_age_and_hosp[d]))
                    end
    
                    out_arr = @inbounds curve_mush(
                        n_days,
                        n_steps_per_day,
                        hosp_curve,
                        time_varying_sample.pr_ICU[j,:],
                        group_param_sample[j]
                    )
    
                    samples_out[p, :] .+= out_arr[1:n_steps_per_day:(n_days * n_steps_per_day), c_ward, s_occupancy]
                    samples_out[p, :] .+= out_arr[1:n_steps_per_day:(n_days * n_steps_per_day), c_postICU_to_death, s_occupancy]
                    samples_out[p, :] .+= out_arr[1:n_steps_per_day:(n_days * n_steps_per_day), c_postICU_to_discharge, s_occupancy]
                end
    
                distance = maximum(abs.(samples_out[p, ward_vec_days] .- true_ward_vec[ward_vec_days]))
    
                if distance < threshold
                    accepted = true
                else
                    samples_out[p, :] = zeros(n_days)
                end

                n_attempts += 1
            end

            attempts[p, t] = n_attempts

            if t == 1
                weights[p, t] = 1
            else
                numer = prod(pdf(prior_pr_hosp, params[p, t]))
                
                denom = 0

                for j in 1:n_particles
                    denom += weights[j, t - 1] * pdf(Normal(params[j, t - 1], 0.025), params[j, t])
                end

                weights[p, t] = numer / denom
            end
        end
    end



    


    return (
        samples_out = samples_out,
        weights = weights,
        params = params,
        attempts = attempts
    )

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