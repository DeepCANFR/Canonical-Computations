import numpy as np

import itertools


def CC_diagram(space_time_diagrams):
    '''
    Produces a canonical computation (CC) diagram from a stack of space-time diagrams.
    The CCs are labeled numerically in the order they are found.
    A dictionary containing containing numeric value - Boolean function (in the form of a binary array)
    is returned so that the specific function can be looked up

    Space-time diagrams should be an array of space-time diagrams 
    with diagram nr on the 0th axis, space on the 1th axis and time over the 3rd axis.
    The configuration at t = 0 should be the starting state
    '''
    
    # CC_labels contains binary strings - numeric label pairs. 
    # The binary string corresponds to a Boolean function on the input set
    CC_labels = {}

    # Contains the numerica value - boolean function pairs
    Boolean_functions = {}

    # The numeric value of the next new Boolean function to be discovered
    next_new_CC_numeric_label = 0

    # Canonical Computation diagram
    CC_diagram = np.zeros((space_time_diagrams.shape[1], space_time_diagrams.shape[2]), 
                          dtype = np.uint64)
    
    for time_coordinate in range(space_time_diagrams.shape[2]):
        for space_coordinate in range(space_time_diagrams.shape[1]):
            
            # Extracts the binary string from a space-time location
            # and converts it to a byte string such that it can be used as a key in a dict
            CC_label = space_time_diagrams[:,space_coordinate, time_coordinate]
            CC_label = CC_label.tobytes()

            # Checks if the CC_label (the boolean function) was previously found
            # Then labels the space-time location with the appropriate numeric label
            if CC_label in CC_labels.keys():
                CC_numeric_label = CC_labels[CC_label]
                CC_diagram[space_coordinate, time_coordinate] = CC_numeric_label
            else:
                # If the current CC_label is novel we add the current next_new_CC_numeric_label
                # to the CC_labels together with its Boolean function
                # We then increase the numeric label by 1
                
                CC_labels[CC_label] = next_new_CC_numeric_label
                CC_diagram[space_coordinate, time_coordinate] = CC_labels[CC_label]
                Boolean_functions[next_new_CC_numeric_label] = space_time_diagrams[:,space_coordinate, time_coordinate]
                next_new_CC_numeric_label += 1
                
            
    
    return CC_diagram, Boolean_functions


def create_CC_labels(input_set_size, CA_states = [0,1]):
    ## Compute all possible boolean functions as byte strings
    all_CCs = [np.array(list(i), dtype = np.uint8).tobytes() for i in itertools.product(CA_states, repeat=input_set_size)]

    # Numerically labels each CC
    CC_labels = {}
    for label, CC in enumerate(all_CCs):
        CC_labels[CC] = label
    return CC_labels

    
def CC_diagram_consistent_labels(space_time_diagrams, CC_labels):
    # WARNING! Memory intensive
    '''
    Produces a canonical computation (CC) diagram from a stack of space-time diagrams.
    The CCs are labeled based on their corresponding binary string, which should be contained in CC_labels. 
    This means that given the same input set, the labels will be identical. 

    Space time diagrams should be an array of space time diagrams 
    with space on the 0th axis, time on the 1st axis and stacked over the 3rd axis.
    The configuration at t = 0 should be the starting state
    '''

    CC_diagram = np.zeros((space_time_diagrams.shape[1], space_time_diagrams.shape[2]),
                                      dtype = np.uint64)
    

    for time_coordinate in range(space_time_diagrams.shape[2]):
        for space_coordinate in range(space_time_diagrams.shape[1]):
           
            CC_label = space_time_diagrams[:,space_coordinate, time_coordinate].astype(np.uint8)
            CC_label = CC_label.tobytes()
            CC_diagram[space_coordinate, time_coordinate] = CC_labels[CC_label]
  
    return CC_diagram