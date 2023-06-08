
import StatsBase
using LinearAlgebra

function post_regularise(particle_collection, weights, rng)
    bandwidth_scale = 0.5


    num_particles = n_particles(particle_collection)
    num_params = 2
    
    h = bandwidth_scale * (4 / (num_particles * (num_params + 2))) ^ (1 / (num_params + 4))


    x = zeros(num_particles, num_params)

    for p in 1:num_particles
        particle_p = particle(particle_collection, p)

        x[p, 1] = particle_p.adj_pr_hosp
        x[p, 2] = particle_p.adj_los

        #x[p, 3] = particle_p.log_ward_importation_rate
        #x[p, 4] = particle_p.log_ward_clearance_rate
    end

    cov_mat = cov(x, StatsBase.Weights(weights))
    a_mat = cholesky(cov_mat)

    std_samples = rand(rng, Normal(), (num_params, num_particles))
    scaled_samples = (a_mat.L * (h * std_samples))'

    particles_out = Vector{pf_state}(undef, num_particles)

    for p in 1:num_particles
        pf_state_old = particle(particle_collection, p)

        particles_out[p] = pf_state(
            pf_state_old.arr_all,

            pf_state_old.adj_pr_hosp + scaled_samples[p, 1],
            pf_state_old.adj_los + scaled_samples[p, 2],

            pf_state_old.log_ward_importation_rate,# + scaled_samples[p, 3],
            pf_state_old.log_ward_clearance_rate,# + scaled_samples[p, 4],

            pf_state_old.case_curve,

            ward_epidemic(
                pf_state_old.epidemic.S, pf_state_old.epidemic.I, pf_state_old.epidemic.Q
            )
        )
    end

    return ParticleCollection(particles_out)
end