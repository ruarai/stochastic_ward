
function curve_mush(
    n_days,
    n_steps_per_day,

    arr,

    case_curve,
    pr_ICU_curve,
    group_params,
    group_delay_samples
)
    n_steps = n_days * n_steps_per_day

    rng = MersenneTwister()

    fill!(arr, 0.0)


    arr[1:n_steps_per_day:n_steps, c_symptomatic, s_transitions] .= case_curve


    for t in 1:n_steps
        for c in 1:def_n_compartments

            # Update the current occupancy value to reflect the value at the last timestep
            # counted at arr[ix(t - 1, c, s_occupancy)]
            # plus/minus any transitions (counted by arr[ix(t, c, s_occupancy)])
            if t > 1
                arr[t, c, s_occupancy] = arr[t, c, s_occupancy] + arr[t - 1, c, s_occupancy]
            end

            # Also increment occupancy by number of inward transitions
            arr[t, c, s_occupancy] = arr[t, c, s_occupancy] + arr[t, c, s_transitions]
        end

        transition_delay(
            c_symptomatic, c_ward,
            arr[t, c_symptomatic, s_transitions],
            t,
            arr,
            group_delay_samples.symptomatic_to_ward,
            n_steps, n_steps_per_day,

            rng
        )

        pr_ICU = pr_ICU_curve[max(t ÷ n_steps_per_day - 5, 1)]

        pr_not_ICU = 1 - pr_ICU
        pr_discharge_given_not_ICU = group_params.pr_ward_to_discharge / (1 - group_params.pr_ward_to_ICU)
        pr_discharge_adj = pr_discharge_given_not_ICU * pr_not_ICU;

        transition_ward_next(
            t, arr,
            pr_discharge_adj, pr_ICU,

            group_delay_samples,
            n_steps, n_steps_per_day,
            
            rng
        )

        transition_ICU_next(
            t, arr,
            group_params,
            group_delay_samples,
            n_steps, n_steps_per_day,

            rng
        )

        transition_delay(
            c_postICU_to_death, c_died_postICU,
            arr[t, c_postICU_to_death, s_transitions],
            t,
            arr,
            group_delay_samples.postICU_to_death,
            n_steps, n_steps_per_day,

            rng
        )
        transition_delay(
            c_postICU_to_discharge, c_discharged_postICU,
            arr[t, c_postICU_to_discharge, s_transitions],
            t,
            arr,
            group_delay_samples.postICU_to_discharge,
            n_steps, n_steps_per_day,

            rng
        )



    end

    return arr
end
