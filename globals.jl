global def_n_compartments::Int32 = 11
global def_n_slots::Int32 = 2
global def_n_age_groups::Int32 = 9

global def_n_ward_epidemic::Int32 = 4
global ward_steady_state_size::Int32 = 50


global age_groups = [
    "0-9", "10-19", "20-29", "30-39", "40-49",
    "50-59", "60-69", "70-79", "80+"
]

global c_symptomatic::Int32 = 1

global c_ward::Int32 = 2

global c_discharged_ward::Int32 = 3
global c_died_ward::Int32 = 4

global c_ICU::Int32 = 5
global c_discharged_ICU::Int32 = 6
global c_died_ICU::Int32 = 7

global c_postICU_to_discharge::Int32 = 7
global c_postICU_to_death::Int32 = 8

global c_discharged_postICU::Int32 = 9
global c_died_postICU::Int32 = 10


global s_transitions::Int32 = 1
global s_occupancy::Int32 = 2