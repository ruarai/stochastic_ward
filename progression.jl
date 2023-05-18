

function transition_delay(
    c_from, c_to,
    n_to_transition,
    t,

    arr,
    delay_samples,
    n_steps, n_steps_per_day,

    rng
)

    for i in 1:n_to_transition
        d = sample(rng, delay_samples)

        t_set = d + t + 1

        if t_set > n_steps
            continue
        end

        arr[t_set, c_to, s_transitions] += 1
        arr[t_set, c_from, s_occupancy] -= 1

    end
end


function transition_ward_next(
    t,
    arr,

    pr_ward_to_discharge,
    pr_ward_to_ICU,
    group_delay_samples,
    n_steps, n_steps_per_day,

    rng
)
    n_to_transition = arr[t, c_ward, s_transitions]

    n_to_discharge = 0
    n_to_death = 0
    n_to_ICU = 0

    for i in 1:n_to_transition
        sample_a = rand(rng)

        if sample_a < pr_ward_to_discharge
            n_to_discharge += 1
        elseif sample_a < pr_ward_to_discharge + pr_ward_to_ICU
            n_to_ICU += 1
        else
            n_to_death += 1
        end
    end

    transition_delay(
        c_ward, c_discharged_ward,
        n_to_discharge,
        t, arr,
        group_delay_samples.ward_to_discharge,
        n_steps, n_steps_per_day,

        rng
    )
    transition_delay(
        c_ward, c_died_ward,
        n_to_death,
        t, arr,
        group_delay_samples.ward_to_death,
        n_steps, n_steps_per_day,

        rng
    )
    transition_delay(
        c_ward, c_ICU,
        n_to_ICU,
        t, arr,
        group_delay_samples.ward_to_ICU,
        n_steps, n_steps_per_day,

        rng
    )

end

function transition_ICU_next(
    t,
    arr,

    group_params,
    group_delay_samples,
    n_steps, n_steps_per_day,

    rng
)
    n_to_transition = arr[t, c_ICU, s_transitions]

    n_to_discharge = 0
    n_to_death = 0
    n_to_postICU_death = 0
    n_to_postICU_discharge = 0

    for i in 1:n_to_transition
        sample_a = rand(rng)

        if sample_a < group_params.pr_ICU_to_discharge
            n_to_discharge += 1
        elseif sample_a < group_params.pr_ICU_to_discharge + group_params.pr_ICU_to_postICU
            sample_b = rand(rng)

            if sample_b < group_params.pr_postICU_to_death
                n_to_postICU_death += 1
            else
                n_to_postICU_discharge += 1
            end
        else
            n_to_death += 1
        end
    end

    transition_delay(
        c_ICU, c_discharged_ICU,
        n_to_discharge,
        t, arr,
        group_delay_samples.ICU_to_discharge,
        n_steps, n_steps_per_day,

        rng
    )

    transition_delay(
        c_ICU, c_died_ICU,
        n_to_death,
        t, arr,
        group_delay_samples.ICU_to_death,
        n_steps, n_steps_per_day,

        rng
    )

    transition_delay(
        c_ICU, c_postICU_to_death,
        n_to_postICU_death,
        t, arr,
        group_delay_samples.ICU_to_postICU,
        n_steps, n_steps_per_day,

        rng
    )

    transition_delay(
        c_ICU, c_postICU_to_discharge,
        n_to_postICU_discharge,
        t, arr,
        group_delay_samples.ICU_to_postICU,
        n_steps, n_steps_per_day,

        rng
    )
end
