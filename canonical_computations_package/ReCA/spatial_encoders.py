import numpy as np

def k_to_n_encoder(input_set, encoded_input_size, locations):
    '''
    
    '''
    if input_set.shape[1] != len(locations):
        raise Exception("The length of input examples must equal the number of locations given")
            
    encoded_input_set = np.zeros((input_set.shape[0], encoded_input_size))
    
    for input_set_index in range(input_set.shape[0]):
        for encoding_index, encoding_location in enumerate(locations):
            encoded_input_set[input_set_index, encoding_location] = input_set[input_set_index, encoding_index]
    return encoded_input_set