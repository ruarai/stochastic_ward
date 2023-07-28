function get_ward_outbreak_occupancy(model_state, d)
    return model_state.epidemic.outbreak_occupancy[d]
end

function get_ward_progression_occupancy(model_state, t)
    arr_all = model_state.arr_all

    if t == 1
        return 0
    else
        return sum(arr_all[:, t - 1, c_ward, s_occupancy]) + 
            sum(arr_all[:, t - 1, c_postICU_to_death, s_occupancy]) + 
            sum(arr_all[:, t - 1, c_postICU_to_discharge, s_occupancy]) 
    end
end

function get_total_ward_occupancy(model_state, t, d)
    return get_ward_progression_occupancy(model_state, t) + get_ward_outbreak_occupancy(model_state, d)
end


function get_total_ICU_occupancy(model_state, t)
    arr_all = model_state.arr_all
    if t == 1
        return 0
    else
        return sum(arr_all[:, t - 1, c_ICU, s_occupancy])
    end
end