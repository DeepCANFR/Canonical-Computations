import numpy as np
from numba import njit

@njit
def find_shift_cycles(CC_diagram):
    '''
    Function used to detect if a diagram contains spatially shifted cycles.
    I.e if the current configuration at some t is identical to some test configuration at time  t + t2, but shifted
    spatially
    '''
  
    # iterates over time 
    for t in range(CC_diagram.shape[1]):
        # iterates over remaining timepoints after the current t
        for t2 in range(t+1, CC_diagram.shape[1]):
            # nr of cells to shift the test configuration
            for r in range(CC_diagram.shape[0]):
                current_configuration = CC_diagram[:,t]
                test_configuration = CC_diagram[:,t2]
                # shifts test configuration
                test_configuration = np.roll(test_configuration, r)
                
                # Checks if shifted test configuration is identical to the current configuration
                if np.sum(current_configuration == test_configuration) == CC_diagram.shape[0]:
                    # Info conferning cycle 
                    # [cycle start, cycle end, cycle length, nr of cells shifted]
                    cycle = np.array([t,t2,t2-t, r])
                     
                    return cycle
     
    # If no cycle is found a vector of 0s is returned
    # This vector could not be produced if a cycle was found.
    cycle = np.array([0,0,0,0])
    return cycle
    
@njit
def find_cycles(CC_diagram):
    """
    Finds non-shifted cycles in CC diagrams
    """
  
    for t in range(CC_diagram.shape[1]):
        for t2 in range(t+1, CC_diagram.shape[1]):

            current_configuration = CC_diagram[:,t]
            test_configuration = CC_diagram[:,t2]
  
            if np.sum(current_configuration == test_configuration) == CC_diagram.shape[0]:
                cycle = np.array([t,t2,t2-t])
                return cycle

    cycle = np.array([0,0,0])
    return cycle
            