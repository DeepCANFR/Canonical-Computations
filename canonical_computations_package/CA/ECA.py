import numpy as np
from numba import njit

@njit
def ECA_rule_mapping(wolfram_code):
    """
    Creates a vector of the states each neighbourhood configuration in an ECA 
    produces according to Wolfram's ordering of neighbourhoods
    """
    rule_mapping = (wolfram_code  & (2**np.arange(8)) != 0)[::-1]
    return rule_mapping

@njit
def create_rule_array(wolfram_code, size):
    """
    Creates an array that implements an ECA rule given by a Wolfram code.
    The use of a rule array makes it possible to use cupy instead of numpy
    for GPU parallelization of run_ECA
    """

    # Produce the rule transition mapping for the rule given by the wolfram code
    rule_mapping = ECA_rule_mapping(wolfram_code)

    # Possible ECA neighbourhood configurations ordered to fit the rule mapping above
    neighbourhood_states = np.array([
                                [1,1,1],
                                [1,1,0],
                                [1,0,1],
                                [1,0,0],
                                [0,1,1],
                                [0,1,0],
                                [0,0,1],
                                [0,0,0]
                            ],
                            dtype = np.uint8)
    
    # Because of the implementation of run_ECA we only need entries in
    # the rule array for neighbourhood configurations that transition to 1
    nr_of_transitions_to_1 = np.sum(rule_mapping)

    # Create rule array
    rule_array = np.zeros((size, 3, nr_of_transitions_to_1), dtype = np.uint8)
    rule_array[:,:,:] = neighbourhood_states[rule_mapping.astype(np.bool_),:].T

    return rule_array


@njit
def run_ECA(wolfram_code, initial_configuration, simulation_length):
    """
    Creates a space-time diagram for a given ECA rule, initial configuration and simulation length
    """

    size = initial_configuration.shape[0]
    # Initiate empty space-time diagram
    space_time_diagram = np.zeros((size, simulation_length), dtype = np.uint8)

    # set t= 0 to initial configuration
    space_time_diagram[:,0] = initial_configuration
    

    rule_array = create_rule_array(wolfram_code, size)
    rule_configuration_array = np.zeros(rule_array.shape, dtype = np.uint8)
    
    neighbourhood_configuration = np.zeros((size, 3), dtype = np.uint8)
    for t in range(1,simulation_length):
        
        # setup current neighbourhood configuration
        neighbourhood_configuration[:,0] = np.roll(space_time_diagram[:,t-1], -1)
        neighbourhood_configuration[:,1] = space_time_diagram[:,t-1]
        neighbourhood_configuration[:,2] = np.roll(space_time_diagram[:,t-1], 1)  
        
        # iterate through rule array to look for match with neighbourhood configuration
        # If one of the neighbourhoods matches, the 2nd axis for that cell will all be 1s
        for i0 in range(rule_array.shape[2]):
            rule_configuration_array[:,:,i0] = neighbourhood_configuration == rule_array[:,:,i0]
            
        # Creates a 2d array of 1s, the first axis is space the second is the number of transitions to 1
        next_configuration = np.ones((rule_configuration_array.shape[0], rule_configuration_array.shape[2]),
                                      dtype = np.uint8)
        
        # Masks out transitions to 1 such that if the neighbourhood did transition to 1
        # the spatial locations has a single 1 in the next_configuration array
        for i0 in range(rule_configuration_array.shape[1]):
            next_configuration *=  rule_configuration_array[:,i0,:]
            
        # Collapse next_configuration to 1D and input it to the space-time diagram
        space_time_diagram[:,t] = np.sum(next_configuration,1)
        
    return space_time_diagram