


function repeat_sims()


    group_param = group_parameters(
        0.7, 0.2,

        0.1, 0.8,

        0.1,

        Gamma(4, 2),

        Gamma(4, 2),
        Gamma(4, 2),
        Gamma(4, 2),

        Gamma(4, 2),
        Gamma(4, 2),
        Gamma(4, 2),

        Gamma(4, 2),
        Gamma(4, 2)
    )


    n_days = 100
    n_steps_per_day = 4

    hosp_curve = rand(Poisson(12), n_days)

    pr_ICU = logistic.(rand(Normal(-1, 1), n_days))


    group_samples = make_delay_samples(group_param, 512, n_steps_per_day)

    
    Threads.@threads for i in 1:10000

        arr = zeros(Int32, n_days * n_steps_per_day, def_n_compartments, def_n_slots)
        @inbounds curve_mush(
            n_days,
            n_steps_per_day,

            arr,

            hosp_curve,
            pr_ICU,
            group_param,
            group_samples
        )
    end

end



@time repeat_sims()


hosp_curve = rand(Poisson(12), 100)

pr_ICU = logistic.(rand(Normal(-1, 1), 100))


group_param = group_parameters(
    0.7, 0.2,

    0.1, 0.8,

    0.1,

    Gamma(4, 2),

    Gamma(4, 2),
    Gamma(4, 2),
    Gamma(4, 2),

    Gamma(4, 2),
    Gamma(4, 2),
    Gamma(4, 2),

    Gamma(4, 2),
    Gamma(4, 2)
)

group_samples = make_delay_samples(group_param, 512, 4)

x = @allocated curve_mush(
    100,
    4,
    hosp_curve,
    pr_ICU,
    group_param,
    group_samples
)