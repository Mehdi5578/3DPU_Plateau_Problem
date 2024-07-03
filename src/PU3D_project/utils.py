import numpy as np

def compute_absolute_phase_gradients(wrapped_phase, unwrapped_phase):

    gradient_x_unwrapped = np.diff(unwrapped_phase, axis=0, append=unwrapped_phase[0:1,:,:])
    gradient_y_unwrapped = np.diff(unwrapped_phase, axis=1, append=unwrapped_phase[:,0:1,:])
    gradient_z_unwrapped = np.diff(unwrapped_phase, axis=2, append=unwrapped_phase[:,:,0:1])
    
    gradient_x_wrapped = np.diff(wrapped_phase, axis=0, append=wrapped_phase[0:1,:,:])
    gradient_y_wrapped = np.diff(wrapped_phase, axis=1, append=wrapped_phase[:,0:1,:])
    gradient_z_wrapped = np.diff(wrapped_phase, axis=2, append=wrapped_phase[:,:,0:1])
    
    
    abs_diff_x = np.abs(gradient_x_unwrapped - gradient_x_wrapped)
    abs_diff_y = np.abs(gradient_y_unwrapped - gradient_y_wrapped)
    abs_diff_z = np.abs(gradient_z_unwrapped - gradient_z_wrapped)

    total_diff = np.count_nonzero(abs_diff_x) + np.count_nonzero(abs_diff_y) + np.count_nonzero(abs_diff_z)
    
    return total_diff
