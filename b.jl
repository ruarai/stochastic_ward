

group_params = group_parameters(
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

n_days = 200
n_steps_per_day = 4

group_samples = make_delay_samples(group_params, 512, n_steps_per_day)

hosp_curve = rand(Poisson(4), n_days)

pr_ICU_curve = logistic.(rand(Normal(-1, 1), n_days))


x = curve_mush(
    n_days,
    n_steps_per_day,
    zeros(n_days * n_steps_per_day, def_n_compartments, def_n_slots),
    hosp_curve,
    pr_ICU_curve,
    group_params,
    group_samples
)




# not considering post-ICU
ward_occ_true = x[:, c_ward, s_occupancy]

struct pf_state
    arr::Array{Int32, 3}
    t::Int32
end

x_true = pf_state(
    copy(x),
    1
)


n_steps::Int32 = n_days * n_steps_per_day


arr = zeros(Int32, n_steps, def_n_compartments, def_n_slots)

arr[1:n_steps_per_day:n_steps, c_symptomatic, s_transitions] .= hosp_curve


function  g(xt, ut, xt1, yt1)
    x_occ = xt1.arr[xt1.t - 1, c_ward, s_occupancy]
    return pdf(Normal(yt1, 5), x_occ)
end


rng = MersenneTwister()
function f(xt, ut, rng)

    t = xt.t
    arr = copy(xt.arr)

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
        group_samples.symptomatic_to_ward,
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

        group_samples,
        n_steps, n_steps_per_day,
        
        rng
    )

    transition_ICU_next(
        t, arr,
        group_params,
        group_samples,
        n_steps, n_steps_per_day,

        rng
    )

    transition_delay(
        c_postICU_to_death, c_died_postICU,
        arr[t, c_postICU_to_death, s_transitions],
        t,
        arr,
        group_samples.postICU_to_death,
        n_steps, n_steps_per_day,

        rng
    )
    transition_delay(
        c_postICU_to_discharge, c_discharged_postICU,
        arr[t, c_postICU_to_discharge, s_transitions],
        t,
        arr,
        group_samples.postICU_to_discharge,
        n_steps, n_steps_per_day,

        rng
    )

    

    return pf_state(
        arr,
        t + 1
    )
end

using ParticleFilters

m = ParticleFilterModel{pf_state}(f, g);

n_particles = 1000

fil = BootstrapFilter(m, n_particles);

b0 = ParticleCollection([pf_state(copy(arr), 1) for i in 1:n_particles]);

us = zeros(n_steps)

occ_at_time = zeros(n_steps, n_particles)

for t in 1:n_steps
    b0 = update(fil, b0, 0.0, ward_occ_true[t]);

    for p in 1:n_particles
        occ_at_time[t, p] = particle(b0, p).arr[t, c_ward, s_occupancy]
    end
end


plot(occ_at_time[:,1:100])
plot!(ward_occ_true, lc = "black", legend = false)


plot(particle(b0, 1).arr[:, c_ward, s_occupancy])
for i in 2:400
    plot!(particle(b0, i).arr[:, c_ward, s_occupancy])
end

plot!(ward_occ_true, lc = "black", legend = false)
