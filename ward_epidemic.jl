

struct ward_epidemic
    outbreak_occupancy::Vector{Float64}
end

function run_ward_epidemic!(epidemic, n_days, case_curve, log_importation_rate, log_clearance_rate)
    importation_rate_standardised = exp(log_importation_rate) / def_n_ward_epidemic

    outbreak_occupancy_wards = zeros(n_days, def_n_ward_epidemic)
    
    for i in 1:def_n_ward_epidemic
        S = ward_steady_state_size
        E = 0
        I = 0
        Q = 0
    
        d = 1
        while true
            if I + Q == 0
                S = ward_steady_state_size
            end
    
            rate_new_I = 50.0 * importation_rate_standardised
            rate_S_to_E = S * I * 0.9 / (I + S + 1e-10)
            rate_E_to_I = E * 0.3
            rate_I_to_Q = I * 0.5
            rate_clearance_Q = Q * exp(log_clearance_rate)
    
            rate_sum = sum(rate_new_I + rate_S_to_E + rate_E_to_I + rate_I_to_Q + rate_clearance_Q)
            time_delta = (1 / rate_sum) * log(1 / rand())
    
            if time_delta > 1
                d += 1
            else
                d += time_delta
    
                rand_event_sample = rand() * rate_sum
    
                if rand_event_sample < rate_new_I
                    I += 1
                elseif rand_event_sample < rate_new_I + rate_S_to_E
                    S -= 1
                    E += 1
                elseif rand_event_sample < rate_new_I + rate_S_to_E + rate_E_to_I
                    E -= 1
                    I += 1
                elseif rand_event_sample < rate_new_I + rate_S_to_E + rate_E_to_I + rate_I_to_Q
                    I -= 1
                    Q += 1
                else
                    Q -= 1
                end
    
            end
    
            if d > n_days
                break
            end
    
            outbreak_occupancy_wards[floor(Int, d), i] = Q
        end
    
        outbreak_occupancy_wards[n_days, i] = Q
    end

    for d in 1:n_days
        epidemic.outbreak_occupancy[d] = sum(outbreak_occupancy_wards[d, :])
    end
end