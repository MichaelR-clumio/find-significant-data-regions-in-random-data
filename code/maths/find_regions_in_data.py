# Example of data in inputs
"""
{
    "num_points": 1000,
    "rnd_max": 11,
    "rnd_min": 0,
    "window": [
          [100,125,"p",30],
          [200,250,"m",3],
          [330,305,"p",50],
          [335,360,"m",1.5],
          [400,425,"p",10],
          [600,617,"m",2],
          [731,777,"p",56],
          [860,960,"m",4]
        ]
}
"""

# other
import json
import os
import numpy as np
import matplotlib.pyplot as plt

def find_0_boundary(prev_slice,post_slice):
    # Inside of the rough boundaries determined by the average of average use
    # first and second order approximation to determine actual boundaries of significant region

    # create a slice of data from the peak to the starting boundary
    pre_boundary_index = 0
    # create a slice of data from the peak to the ending boundary
    post_boundary_index = len(post_slice) - 1
    # Find the maximum positive slope value inside this region before the peak point
    max_val_indice_prev = np.argmax(prev_slice)

    # Find the maximum negative slope value inside this region after the peak point
    min_val_indice_post = np.argmin(post_slice)

    # In the slice of data before the peak using the second order approximation find the "zero" point after the max value
    # This is the start point of the region
    abs_prev_slice_n = prev_slice[max_val_indice_prev:]
    if prev_slice[max_val_indice_prev] < 0:
        print("failure condition")
        return False
    else:
        for i in range(len(abs_prev_slice_n)):
            if abs_prev_slice_n[i] < 0:
                pre_boundary_index =  max_val_indice_prev + i
                break

    # In the slice of data after the peak using the second order approximation find the "zero" point after the max negative value
    # This is the end point of the region
    abs_post_slice_n = post_slice[min_val_indice_post:]
    if post_slice[min_val_indice_post] > 0:
        print("failure condition")
        return False, 0, 0
    else:
        for i in range(len(abs_post_slice_n)):
            if abs_post_slice_n[i] > 0:
                post_boundary_index =  min_val_indice_post + i
                break
    return True, pre_boundary_index, post_boundary_index


def find_boundary_m(m_arr, top_peaks,avg_m):
    # Looks at each top peak
    # Using the smoothed data set finds the location before and after a peak where the
    #  average of average data value is less the average value across the entire data set.
    # This creates a rough boundary condition to use in further processing.
    peak_boundaries_m = {}
    sc_top_peaks = top_peaks.copy()
    while np.any(sc_top_peaks):
        cur_peak_indice = np.argmax(sc_top_peaks)
        cur_peak = np.max(sc_top_peaks)
        #print(f"processing peak at {cur_peak_indice} with value {cur_peak}")
        not_found_previous_boundary = True
        not_found_after_boundary = True
        previous_avg_boundary = 0
        next_avg_boundary = len(m_arr) - 1
        for i in range(1,len(m_arr)):
            #if not i == center:
            index_p = int(cur_peak_indice - i)
            index_n = int(cur_peak_indice + i)
            #print(f"index {index_p} at {index_n}")
            if not_found_previous_boundary:
                if index_p == 0:
                    next_avg_boundary = index_p
                    not_found_previous_boundary = False
                elif m_arr[index_p] > avg_m:
                    if sc_top_peaks[index_p] > 0:
                        sc_top_peaks[int(index_p)] = 0
                        #removed = absorb_peak(index_p,removed_peaks)
                        #print(f"removed peak at {index_p}")
                else:
                    previous_avg_boundary = index_p
                    not_found_previous_boundary = False
            if not_found_after_boundary:
                if index_n + 1 == len(m_arr):
                    next_avg_boundary = index_n
                    not_found_after_boundary = False
                elif m_arr[index_n] > avg_m:
                    if sc_top_peaks[index_n] > 0:
                        sc_top_peaks[int(index_n)] = 0
                        # removed = absorb_peak(index_p,removed_peaks)
                        #print(f"removed peak at {index_n}")
                else:
                    next_avg_boundary = index_n
                    not_found_after_boundary = False
            #print(f"flag {not_found_previous_boundary} {not_found_after_boundary} {previous_avg_boundary} {next_avg_boundary}")
            if not (not_found_previous_boundary or not_found_after_boundary):
                peak_boundaries_m[int(cur_peak_indice)] = [int(previous_avg_boundary),int(next_avg_boundary)]
                sc_top_peaks[cur_peak_indice] = 0
                #print(peak_boundaries_m)
                break
        #print(top_peaks)
        #print(sc_top_peaks)
    return peak_boundaries_m


def find_slope(arr_slice,center_point):
    # Looks for points on the smoothed data set where a peak was reached (i.e. slope change direction)

    # Initialize  array to hold results
    size = (len(arr_slice))  # Shape of the array (rows, columns)
    value = 0  # Constant value to fill the array with
    local_slopes = np.full(size, value, dtype=np.double)

    # Look forward and backwards from the center point
    for i in range(center_point):
        index_p = i
        index_n = len(arr_slice) - i - 1
        #print(f" i {i} center {center_point}")
        # Calculate slope from center to the point "i" away from the center
        x = (arr_slice[center_point] - arr_slice[index_p]) / (center_point - index_p)
        y = (arr_slice[index_n] - arr_slice[center_point]) / (index_n - center_point)
        #print(i, x, y)
        # Sum those slopes
        local_slopes[index_p] = x
        local_slopes[index_n] = y
    #print(f" local slopes {local_slopes[0:center_point]}")
    # avg slope before to the center
    start_slope = np.mean(local_slopes[0:center_point])
    # avg slope after the center
    end_slope = np.mean(local_slopes[center_point+1:len(arr_slice)])

    #To calculate overall average get rid of center point
    new_loc_slopes = np.delete(local_slopes, center_point)
    # Average slope for this center point
    avg_slope = np.mean(new_loc_slopes)

    # Check if slope changed direction verses the average slope at the center point
    if start_slope - avg_slope > 0: # steeper slope going up then avg slope
        if end_slope - avg_slope < 0: #steeper fall off then average slope
            return True
    return False

def peak_shaping(list_of_peaks,current_list):
    # Because multiple peak values may be associated with one region
    # Calculate the "top" of a region and reduce many peaks to a central peak
    start_ = current_list[0]
    end_ = current_list[-1]
    #print(start_,end_)
    if len(current_list) == 1:
        cur_max_loc = start_
    else:
        peak_slice = list_of_peaks[start_:end_]
        cur_max_loc = int(np.argmax(peak_slice))
        cur_max_loc += start_
    loc_max = list_of_peaks[cur_max_loc]

    return np.array([start_, end_, loc_max, cur_max_loc])

#def process_last_peak(peak_list):
#    return slope_shaping(peak_list)


def find_peak_shape(list_of_peaks):
    # Because multiple peak values may be associated with one region
    # Calculate the "top" of a region and reduce many peaks to a central peak
    size = ((len(list_of_peaks),4))  # Shape of the array (rows, columns)
    value = np.array([0,0,0,0])  # Constant value to fill the array with

    peak_shape = np.full(size, value)
    #print(peak_shape)
    #peak_shape = np.array
    i = 0
    start_current = 0
    currently_on_peak = False
    new_peak_list = []
    while i < len(list_of_peaks):
        if currently_on_peak:
            if list_of_peaks[i] > 0:
                new_peak_list.append(i)
                i += 1
            else:
                currently_on_peak = False
                last = i -1
                #print(f"last {last}")
                peak_shape[last] = peak_shaping(list_of_peaks,new_peak_list)
                i += 1
        else:
            if list_of_peaks[i] > 0:
                currently_on_peak = True
                new_peak_list = []
                new_peak_list.append(i)
                i += 1
            else:
                i += 1
    if currently_on_peak:
        last = i -2
        #print(f"last {start_current}")

        peak_shape[last] = peak_shaping(list_of_peaks,new_peak_list)

    return peak_shape


def sum_of_sum_numerator_matrix(size):
    # numerator matrix used to quickly calculate the average of averages to smooth the data
    orig_arr = np.arange(1, size + 1)
    inverse_orig = orig_arr[::-1]
    new_arr = np.append(orig_arr, size)
    return np.append(new_arr, inverse_orig)

def sum_of_sum_denominator(size):
    # the denominator used to quickly calculate the average of averages to smooth the data
    d1_arr = np.arange(1, size + 1)
    # print(d1_arr)
    d2 = d1_arr * 2
    d3 = d2 + 1
    return np.sum(d3)

def sum_of_sum_value(loc_arr_slice,s_of_s_numerator_matrix,s_of_s_denominator):
    # For a segment of data determined by the windows size calculate the average of averages for this data
    return np.dot(loc_arr_slice, s_of_s_numerator_matrix.T) / s_of_s_denominator

def create_avg_of_avg_arr(arr,size):
    # Generates a smoothed data set from the original data using an average of averages
    arr_size = (len(arr))  # Shape of the array (rows, columns)
    arr_default = 0  # Constant value to fill the array with
    # Create a rolling average across the entire data set
    mean_arr = np.full(arr_size, arr_default, dtype=np.double)
    for center in range(window_size, len(arr) - size):
        arr_slice = generate_slice(arr, size, center)
        mean_arr[center] = sum_of_sum_value(arr_slice, _numerator, _denominator)
    return mean_arr

def rate_of_change_avg_over_window(arr,size):
    # Calculate an approximation of the slope at each point in the input data set using an average slope between
    # a center point and each point in a region with a region size determined by the window size
    # Returned value in a new np array
    arr_size = (len(arr))  # Shape of the array (rows, columns)
    arr_default = 0  # Constant value to fill the array with

    slope_arr = np.full(arr_size, arr_default, dtype=np.double)
    for center in range(window_size, len(arr) - size):
        arr_slice = generate_slice(arr, size, center)
        #start = center - size
        #end = center + size + 1
        #arr_slice = arr[start:end]
        #print(arr_slice)
        total_value = 0
        for i in range(len(arr_slice)):
            if not i == size:
                index_p = i
                index_n = len(arr_slice) - i - 1
                distance = size - i

                val_p = (arr_slice[size] - arr_slice[index_p]) / distance
                val_n = (arr_slice[index_n] - arr_slice[size]) / distance
                #print(f"distance: {distance}, i+ {index_p} i- {index_n} valp {val_p} valn {val_n}")
                total_value = total_value + (val_p + val_n) / 2
        avg_val = total_value / (len(arr_slice) -1)
        slope_arr[center] = avg_val
        #print(avg_val)
    return slope_arr

def find_peaks(arr,size):
    arr_size = (len(arr))  # Shape of the array (rows, columns)
    arr_default = 0  # Constant value to fill the array with
    peak_arr = np.full(arr_size, arr_default, dtype=np.double)
    for center in range(window_size, len(arr) - size):
        arr_slice = generate_slice(arr, size, center)
        p_flag = find_slope(arr_slice, size)
        if p_flag:
            peak_arr[center] = arr[center]
    return peak_arr

def generate_slice(loc_data_arr,size,center_point):
    # returns a new numpy slice of the original data set
    start = (center_point - size)
    stop = (center_point + size + 1)
    #print(f" gen slice {start} {stop}")
    loc_arr_slice = loc_data_arr[start : stop]
    return loc_arr_slice

def pick_peaks(m_arr,size,factor):
    # Find a single peak value representing each major region

    m_peak_arr = find_peaks(m_arr, size) # Start with all peaks
    ps1 = find_peak_shape(m_peak_arr)  # returns 2 dimension index [start,stop,peak value,index of the max peak]
    non_zero_indices = np.any(ps1, axis=(1))  # mask of locations that are peaks
    std_of_peaks = np.std(ps1[non_zero_indices][:, 2])  # get standard deviating of the value of the peaks
    median_of_peaks = np.median(ps1[non_zero_indices][:, 2]) # get median value of the value of the peaks
    top_condition = ps1[non_zero_indices][:, 2] > median_of_peaks + (factor * std_of_peaks) # look for peaks where the value is <factor> times the standard deviation above the median
    top_values = ps1[non_zero_indices][np.where(top_condition)]  # return an array with just the top peaks filled in
    arr_size = (len(old_data_arr))  # Shape of the array (rows, columns)
    arr_default = 0  # Constant value to fill the array with

    top_peak_arr = np.full(arr_size, arr_default, dtype=np.double)
    for peak_shape in top_values:
        i = peak_shape[3]
        v = peak_shape[2]
        top_peak_arr[i] = v
    # Return an array of values showing all top peaks
    return top_peak_arr

# Start main
if __name__ == '__main__':
    # Load input data
    input_file = "points_inputs.json"
    # if you need to check where the input file is to be loaded from uncomment below
    #print(os.getcwd())
    # Read JSON data from a file
    with open(input_file, 'r') as f:
        input_data = json.load(f)
    num_points = input_data.get('num_points',0)
    rnd_max = input_data.get('rnd_max',0)
    rnd_min = input_data.get('rnd_min',0)
    windows = input_data.get('window',[])

    # Generate random data
    old_data_arr = np.random.randint(rnd_min, rnd_max, num_points, dtype=np.int64)
    # if data windows for create peak data were defined adjust the data set
    for window in windows:
        if window[2] == "p":
            # Adding a value to the data set
            old_data_arr[window[0]:window[1]] += window[3]
        elif window[2] == "m":
            # multiplying data by a factor
            old_data_arr[window[0]:window[1]] = old_data_arr[window[0]:window[1]] * window[3]
        else:
            # Add other mathematical operations here if you wannt
            pass
    # Set other variables that control how data is shaped
    window_size = 5  # How big a set of data is shaped at one time i.e. from center point look at data +- the window size
    factors = [1.75, 1.5, 3, 5, 10] # Set the multiple of the standard deviation to indicate a "significant" peak size for each run
    runs = 3 # Set the number of runs to re-process the data minus the previous peaks

    # Start processing
    _numerator = sum_of_sum_numerator_matrix(window_size) # numerator matrix used to quickly calculate the average of averages to smooth the data
    _denominator = sum_of_sum_denominator(window_size)  # the denominator used to quickly calculate the average of averages to smooth the data

    # Initialize and save values for processing
    boundaries = [] # Used to save the boundaries of each peak region found
    median_val = np.median(old_data_arr)  # Set for first run.  Will be re-calculated with each subsequent run
    current_data_arr = old_data_arr.copy()  # Set for first run.  Will be re-calculated with each subsequent run

    # Process will run the same algorithm multiple times after neutralizing previously found peak regions
    for runs in range(runs):
        next_data_arr = current_data_arr.copy()  # Create the copy the next run will use
        mean_arr = create_avg_of_avg_arr(current_data_arr, window_size)  # Generate the average of averages smoothed data set

        #Save average of averages smoothed data from first run for plotting purposes
        if runs == 0:
            old_mean_arr = mean_arr.copy()

        slope_mean_arr = rate_of_change_avg_over_window(mean_arr, window_size) # 1st order approximation: Generate an average of slopes from the average of average data

        slope_slope_arr = rate_of_change_avg_over_window(slope_mean_arr, window_size) # 2nd order approximation: Generate an average of slopes from the slopes data

        top_peak_arr = pick_peaks(mean_arr, window_size, factors[runs]) # find a peak for each significant region

        avg_mean = np.mean(mean_arr) # calculate the average value across the smoothed average of average values

        # find the approximate boundaries of each significant region based upon first order approximation
        bound_dict = find_boundary_m(mean_arr, top_peak_arr, avg_mean)

        # Define the actual boundaries relative to the original data set
        for key in bound_dict.keys():
            # Use 1st order approximation to set region for next part of the algorithm
            a_slice_b_p = slope_slope_arr[bound_dict[key][0] - window_size * 2: key]
            a_slice_b_n = slope_slope_arr[key:bound_dict[key][1] + window_size * 2]

            # find exact start and end point for each region based upon 2nd order approximation
            flag, start_b, stop_b = find_0_boundary(a_slice_b_p, a_slice_b_n)
            start_b = bound_dict[key][0] - window_size * 2 + start_b
            stop_b = key + stop_b
            # Neutralize this boundary from data set for next run
            next_data_arr[start_b:stop_b + 1] = median_val

            boundaries.append([start_b, stop_b, key])  # Save boundary values
        # Set new values for next run
        current_data_arr = next_data_arr
        median_val = np.median(current_data_arr)

    # Initialize another array to contain only original data inside the regions identified
    arr_size = (len(old_data_arr))  # Shape of the array (rows, columns)
    arr_default = 0  # Constant value to fill the array with
    masked_data_arr = np.full(arr_size, arr_default, dtype=np.double)

    #Populate data set with only significant data
    for boundary in boundaries:
        print(int(boundary[0]), int(boundary[1]), boundary[2])
        masked_data_arr[boundary[0]:boundary[1]] = old_data_arr[boundary[0]:boundary[1]]

    #Plot the data
    x = np.arange(len(old_data_arr))
    # ax1 shows original data
    # ax2 shows original data in found regions only
    # ax3 shows the smoothed average of averages data
    fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, ncols=1)
    ax1.plot(x, old_data_arr, '.-')
    ax2.plot(x, masked_data_arr, '.-')
    ax3.plot(x, old_mean_arr, '.-')

    ax1.set_xlabel("points")

    ax1.grid()
    ax2.grid()
    ax3.grid()

    plt.show()
    fig.savefig('data_plot.png', bbox_inches='tight')
    plt.close(fig)