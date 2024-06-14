# Your ANU ID: u7544620
# Your NAME: Bo Dai
# I declare that this submission is my own work
# [ref: https://www.anu.edu.au/students/academic-skills/academic-integrity ]

# This code simulates the future trajectory of planets in a planetary system according to an intial configuration within an error tolerance. 

import csv
import numpy as np
from scipy.optimize import minimize_scalar
import time

def load_star_or_planet(path_to_file):
    """
    Reading data from tsv file
    :param path_to_file:
    :return: a tuple of a 1d nparray with names, a 2d nparray in which a row stands for a planet or star with name,mass,coordinate in x_axis,coordinate in y_axis,velocity in x_axis,velocity in y_axis
    """
    name_list = []
    attr_list = []   # saving  planets or stars as a dict
    # Open the TSV file for reading
    with open(path_to_file, mode='r',) as tsv_file:
        # Create a CSV reader with tab as the delimiter
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        # Skip the header row
        header_row = next(tsv_reader)
        # Iterate through each row in the TSV file
        for row in tsv_reader:
            # Append each object to the list
            name_list.append(row[0])
            attr_list.append(row[1:])
    return np.array(name_list),np.array(attr_list,np.float64)
def compute_position_formula(position,velocity,time):
    """
    Compute new positions by the parameters
    :param position: 2d numpy array, each row stands for an object's position
    :param velocity: 2d numpy array, each row stands for an object's velocity
    :param time: a floating-point number for the time (in seconds)
    :return:
    """
    return position + velocity * time

def compute_acceleration_formula(positions,masses):
    """
    Compute acceleration by positions and masses
    :param positions: 2d numpy array, each row stands for an object's position
    :param masses: 1d numpy array, each row stands for an object's mass
    :return: a 2d numpy array, each row is an object's acceleration
    """
    G = 6.67e-11 # gravitational constant
    # Initialize a list to store accelerations
    acceleration_list = []
    for i in range(len(positions)):
        F_i = 0  # initialize F_i to 0
        p_i = positions[i]
        for j in range(len(positions)):
            if i != j:
                p_j = positions[j]
                # calculate the distance between 2 objects
                d_i_j = np.linalg.norm(p_i - p_j)
                # calculate the force between Object i and j according to the formula
                F_i_j = (G * masses[i] * masses[j]) / d_i_j ** 2 * (p_j - p_i) / d_i_j
                F_i += F_i_j  # add F_i_j to F_i
        acceleration_list.append(F_i / masses[i])  # Compute the acceleration and append it to list, attrs[i][0] is m_i
    return np.array(acceleration_list)
def compute_position(path_to_file, time):
    """
    Compute the new position of objects after some time(in seconds)
    :param path_to_file: a string for the path to a file storing the configuration of a planetary system
    :param time: a floating-point number for the time (in seconds)
    :return: a 2-d numpy array with each row stands for an object's new position
    """
    # load objects from the file
    _,attrs = load_star_or_planet(path_to_file)
    #  attr[:,[1,2]] are positions, attr[:,[3,4]] as velocities
    # use the formula to calculate new positions
    return compute_position_formula(attrs[:,[1,2]],attrs[:,[3,4]],time)

def compute_acceleration(path_to_file, time):
    """
    Compute acceleration of all objects in the give file at the given time point
    :param path_to_file: a string for the path to a file storing the configuration of a planetary system
    :param time: a floating-point number for the time (in seconds)
    :return: a 2-d numpy array with each row stands for an object's acceleration at the given time point
    """
    # load objects from the file
    _,attrs = load_star_or_planet(path_to_file)
    # compute the new positions
    positions_new = compute_position_formula(attrs[:,[1,2]],attrs[:,[3,4]],time)
    return compute_acceleration_formula(positions_new,attrs[:,0])

def forward_simulation(path_to_file, total_time, num_steps):
    """
    Base on the number of steps and total time, simulate all the objects in the given file
    :param path_to_file: a string as the path to a file storing the configuration of a planetary system
    :param total_time: a floating-point number for the total time (in seconds)
    :param num_steps:an integer n.
    :return:a 2d numpy array with n rows (each row stands for a step) and 2*k columns (each two columns stand for an object's position)
    """
    # initialize a list to store the result
    all_positions_in_steps = []
    # load the initial stats of all objects
    names,attrs = load_star_or_planet(path_to_file)
    positions = attrs[:,[1,2]]
    masses = attrs[:,0]
    velocities = attrs[:,[3,4]]
    for i in range(num_steps):
        # print(f"steps :{i+1}/{num_steps}")
        # calculate positions base on previous positions and velocities
        positions = compute_position_formula(positions,velocities,total_time/num_steps)
        all_positions_in_steps.append(positions.flatten())
        # update velocities according to new positions   new position --> new acceleration --> new velocity
        accelerations = compute_acceleration_formula(positions,masses)
        velocities = velocities + accelerations * (total_time/num_steps)
    return names, np.array(all_positions_in_steps)

def max_accurate_time(path_to_file, epsilon):
    """
    Find the maximum T such that the total error is still smaller than epsilon.
    Dynamically changing step size to speed up the optimization.
    * double the step size if there's still enough residual
    * half the step size if overshooting, and restart from the middle of the last step
    :param path_to_file: a string as the path to a file storing the configuration of a planetary system
    :param epsilon: a pre-defined threshold
    :return: float
    """
    # define the error_function which needs to be maximized
    def error_function(t_conf):
        # the final positions got step by step
        names,good_positions = forward_simulation(path_to_file,t_conf,t_conf)[-1].reshape(-1,2)
        # the final positions got by just one step
        _,approx_positions = forward_simulation(path_to_file,t_conf,1)[-1].reshape(-1,2)
        # calculate Euclidean distance between the two results
        # and sum all up to get the total error
        diffs = good_positions - approx_positions
        total_error = sum([np.linalg.norm(diff) for diff in diffs])
        return total_error
    # the initial t_conf(T) which sets to 2, since there's no difference if set it to 1
    t_conf = 2
    # set intial step size to 1
    step_size = 1
    while(True):
        # get the error of T, and T+1, say f(T) and f(T+1)
        error = error_function(t_conf)
        error_next = error_function(t_conf+1)
        # print(step_size, epsilon - error,t_conf)
        if(error < epsilon) & (error_next>epsilon):
            # if f(T) < epsilon and f(T+1) >= epsilon, it's the optimum
            break
        elif(error < epsilon):
            # if f(T) < epsilon and f(T+1) < epsilon, change double the step size
            step_size *= 2
            # set the next T
            t_conf += int(step_size)
        else:
            # if f(T) >= epsilon, half the step size
            step_size /= 2
            # reduce T by half of the last step
            t_conf -= int(step_size)
    # print("return: ",t_conf,error)
    return t_conf


def orbit_time(path_to_file, object_name):
    """
    Computing orbit time of a given object
    Give a time >> orbit time, then use its period property to approximate the next time, when the distance from the origin is the smallest.
    :param path_to_file: a string for the path to the file storing the configuration of a planetary system
    :param object_name: a string for the name of an object
    :return: a float number
    """
    # load initial configuration
    names,attrs = load_star_or_planet(path_to_file)
    # get the object's index
    index_of_object = list(names).index(object_name)
    # get its initial position
    initial_position = attrs[index_of_object,[1,2]]
    # print([np.linalg.norm(p) for p in attrs[:,[1,2]]])
    # set total time according to object's relative position to (0,0), the values are got from experiment
    if(np.linalg.norm(initial_position) > 1e12):
        total_time = 1e12
    elif(np.linalg.norm(initial_position) < 1e11):
        total_time = 1e8
    else:
        total_time = 1e9
    # number of steps
    num_steps = 10000
    # get all objects' trajectory
    _,trajectories = forward_simulation(path_to_file, total_time=total_time, num_steps=num_steps)
    # store it in a dict
    name_vs_trajectory = {}
    for i in range(len(names)):
        name_vs_trajectory[names[i]] = trajectories[:,[2*i,2*i+1]]
    # get the object's trajectory
    trajectory = name_vs_trajectory[object_name]
    # calculate the distance between the initial position and all positions
    distances = [np.linalg.norm(initial_position-p) for p in trajectory]
    ## uncomment it for experiment purpose
    # plt.plot(distances)
    # # Add labels and a title
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    # plt.title("Plot of distances")
    # plt.show()
    # chose the first position which is smaller than the position after first step, using the periodical attribute of the distances
    return np.where(distances < distances[0])[0][0] * total_time/num_steps


################################################################################
#                  VISUALISATION
################################################################################

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
import numpy as np


def plot_configuration(xpositions, ypositions, object_names=[]):
    """
        Plot the planetary system
        xpositions: sequence of x-coordinates of all objects
        ypositions: sequence of y-coordinates of all objects
        object_names: (optional) names of the objects
    """

    fig, ax = plt.subplots()

    marker_list = ['o', 'X', 's']
    color_list = ['r', 'b', 'y', 'm']
    if len(object_names) == 0:
        object_names = list(range(len(xpositions)))

    for i, label in enumerate(object_names):
        ax.scatter(xpositions[i],
                   ypositions[i],
                   c=color_list[i % len(color_list)],
                   marker=marker_list[i % len(marker_list)],
                   label=object_names[i],
                   s=70)
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.grid()

    plt.xlabel("x-coordinate (meters)")
    plt.ylabel("y-coordinate (meters)")
    plt.tight_layout()
    plt.show()


def visualize_forward_simulation(table, num_steps_to_plot, object_names=[]):
    """
        visualize the results from forward_simulation
        table: returned value from calling forward_simulation
        num_steps_to_plot: number of selected rows from table to plot
        object_names: (optional) names of the objects
    """
    table = np.array(table)
    if len(object_names) == 0:
        object_names = list(range(len(table[0]) // 2))

    assert len(object_names) == len(table[0]) // 2

    fig = plt.figure()
    num_objects = len(table[0]) // 2
    xmin = min(table[:, 0])
    xmax = max(table[:, 0])
    ymin = min(table[:, 1])
    ymax = max(table[:, 1])
    for i in range(1, num_objects):
        xmin = min(xmin, min(table[:, i * 2]))
        xmax = max(xmax, max(table[:, i * 2]))
        ymin = min(ymin, min(table[:, (i * 2 + 1)]))
        ymax = max(ymax, max(table[:, (i * 2 + 1)]))

    ax = plt.axes(xlim=(xmin, 1.2 * xmax), ylim=(ymin, 1.2 * ymax))

    k = len(table[0]) // 2

    lines = []
    for j in range(1, k):  # Here we are assuming that the first object is the star
        line, = ax.plot([], [], lw=2, label=object_names[j])
        line.set_data([], [])
        lines.append(line)

    N = len(table)

    def animate(i):
        print(i)
        step_increment = N // num_steps_to_plot
        for j in range(1, k):  # Here we are assuming that the first object is the star
            leading_object_trajectories = table[0:i * step_increment]
            x = [ts[2 * j] for ts in leading_object_trajectories]
            y = [ts[2 * j + 1] for ts in leading_object_trajectories]
            lines[j - 1].set_data(x, y)
        return lines

    fig.legend()
    plt.grid()
    matplotlib.rcParams['animation.embed_limit'] = 1024
    anim = FuncAnimation(fig, animate, frames=num_steps_to_plot, interval=20, blit=False)
    plt.show()
    return anim

# trajectories = forward_simulation("solar_system.tsv", 31536000.0*100, 10000)
#objects = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
# animation = visualize_forward_simulation(trajectories, 200, objects)

## Un-comment the lines below to show an animation of
## the planets in the solar system during the next 100 years,
## using 10000 time steps, with only 200 equispaced time steps
## out of these 10000 steps in the whole [0,T] time interval
## actually plotted on the animation
names,object_trajectories=forward_simulation("solar_system.tsv", 31536000.0*100, 10000)
animation=visualize_forward_simulation(object_trajectories, 200,object_names=names)
animation.save("solar_system.gif")
## Un-comment the lines below to show an animation of
## the planets in the TRAPPIST-1 system during the next 20 DAYS,
## using 10000 time steps, with only 200 equispaced time steps
## out of these 10000 steps in the whole [0,T] time interval
## actually plotted on the animation
names,object_trajectories=forward_simulation("trappist-1.tsv", 86400.0*20, 10000)
animation=visualize_forward_simulation(object_trajectories, 200, object_names=names)
animation.save("trappist-1.gif")

################################################################################
#               DO NOT MODIFY ANYTHING BELOW THIS POINT
################################################################################

def test_compute_position():
    '''
    Run tests of the forward_simulation function.
    If all tests pass you will just see "all tests passed".
    If any test fails there will be an error message.
    NOTE: passing all tests does not automatically mean that your code is correct
    because this function only tests a limited number of test cases.
    '''
    position = np.array(compute_position("solar_system.tsv", 86400))
    truth = np.array([[0.00000000e+00, 0.00000000e+00],
                      [4.26808000e+10, 2.39780800e+10],
                      [-1.01015040e+11, -3.75684800e+10],
                      [-1.13358400e+11, -9.94612800e+10],
                      [3.08513600e+10, -2.11534304e+11],
                      [2.11071360e+11, -7.44638848e+11],
                      [6.54704160e+11, -1.34963798e+12],
                      [2.37964662e+12, 1.76044582e+12],
                      [4.39009072e+12, -8.94536896e+11]])
    assert len(position) == len(truth)
    for i in range(0, len(truth)):
        assert len(position[i]) == len(truth[i])
        if np.linalg.norm(truth[i]) == 0.0:
            assert np.linalg.norm(position[i] - truth[i]) < 1e-6
        else:
            assert np.linalg.norm(position[i] - truth[i]) / np.linalg.norm(truth[i]) < 1e-6
    print("all tests passed")


def test_compute_acceleration():
    '''
    Run tests of the compute_acceleration function.
    If all tests pass you will just see "all tests passed".
    If any test fails there will be an error message.
    NOTE: passing all tests does not automatically mean that your code is correct
    because this function only tests a limited number of test cases.
    '''
    acceleration = np.array(compute_acceleration("solar_system.tsv", 10000))
    truth = np.array([[3.42201832e-08, -2.36034356e-07],
                      [-4.98530431e-02, -2.26599078e-02],
                      [1.08133552e-02, 3.71768441e-03],
                      [4.44649916e-03, 3.78461924e-03],
                      [-3.92422837e-04, 2.87361538e-03],
                      [-6.01036812e-05, 2.13176213e-04],
                      [-2.58529454e-05, 5.32663462e-05],
                      [-1.21886258e-05, -9.01929841e-06],
                      [-6.48945783e-06, 1.32120968e-06]])
    assert len(acceleration) == len(truth)
    for i in range(0, len(truth)):
        assert len(acceleration[i]) == len(truth[i])
        assert np.linalg.norm(acceleration[i] - truth[i]) / np.linalg.norm(truth[i]) < 1e-6
    print("all tests passed")


def test_forward_simulation():
    '''
    Run tests of the forward_simulation function.
    If all tests pass you will just see "all tests passed".
    If any test fails there will be an error message.
    NOTE: passing all tests does not automatically mean that your code is correct
    because this function only tests a limited number of test cases.
    '''
    trajectories = forward_simulation("solar_system.tsv", 31536000.0 * 100, 10)

    known_position = np.array([[1.63009260e+10, 4.93545018e+09],
                               [-8.79733713e+13, 1.48391575e+14],
                               [3.54181417e+13, -1.03443654e+14],
                               [5.85930535e+13, -7.01963073e+13],
                               [7.59849728e+13, 1.62880599e+13],
                               [2.89839690e+13, 1.05111979e+13],
                               [3.94485026e+12, 6.29896920e+12],
                               [2.84544375e+12, -3.06657485e+11],
                               [-4.35962396e+12, 2.04187940e+12]])

    rtol = 1.0e-6
    last_position = np.array(trajectories[-1]).reshape((-1, 2))
    for j in range(len(last_position)):
        x = last_position[j]
        assert np.linalg.norm(x - known_position[j]) / np.linalg.norm(known_position[j]) < rtol, "Test Failed!"

    print("all tests passed")


def test_max_accurate_time():
    '''
    Run tests of the max_accurate_time function.
    If all tests pass you will just see "all tests passed".
    If any test fails there will be an error message.
    NOTE: passing all tests does not automatically mean that your code is correct
    because this function only tests a limited number of test cases.
    '''
    assert max_accurate_time('solar_system.tsv', 1) == 5.0
    assert max_accurate_time('solar_system.tsv', 1000) == 163.0
    assert max_accurate_time('solar_system.tsv', 100000) == 1632.0
    print("all tests passed")


def test_orbit_time():
    '''
    Run tests of the orbit_time function.
    If all tests pass you will just see "all tests passed".
    If any test fails there will be an error message.
    NOTE: passing all tests does not automatically mean that your code is correct
    because this function only tests a limited number of test cases.
    '''

    # accepting error of up to 10%
    assert abs(orbit_time('solar_system.tsv', 'Mercury') / 7211935.0 - 1.0) < 0.1
    assert abs(orbit_time('solar_system.tsv', 'Venus') / 19287953.0 - 1.0) < 0.1
    assert abs(orbit_time('solar_system.tsv', 'Earth') / 31697469.0 - 1.0) < 0.1
    assert abs(orbit_time('solar_system.tsv', 'Mars') / 57832248.0 - 1.0) < 0.1
    print("all tests passed")
