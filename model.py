import matplotlib.pyplot as plt
import numpy as np


def draw_plot(math_scores, reading_scores, a=None, b=None):
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(math_scores, reading_scores, color="blue", alpha=0.7, edgecolor="black")
    if a is not None and b is not None:
        # Generate points for the regression line
        line_x = np.linspace(math_scores.min(), math_scores.max(), 100)
        line_y = a * line_x + b
        plt.plot(
            line_x, line_y, color="red", linewidth=2, label=f"y = {a:.2f}x + {b:.2f}"
        )

    # Set axis limits to [0, 100]
    plt.xlim(0, 100)
    plt.ylim(0, 100)

    # Adding titles and labels
    plt.title("Math Scores vs Reading Scores")
    plt.xlabel("Math Scores")
    plt.ylabel("Reading Scores")
    plt.grid(True)

    # Show the plot
    plt.show()


def get_points(path: str):
    points = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=(5, 6))

    return points


def compute_error(a: float, b: float, points: np.ndarray) -> float:
    """
    Computes the mean squared error for a linear fit.

    Parameters:
        a (float): Slope of the line.
        b (float): Intercept of the line.
        points (np.ndarray): A 2D array where each row represents (math, read).

    Returns:
        float: The mean squared error.
    """

    return ((points[:, 1] - (a * points[:, 0] + b)) ** 2).mean()


def gradient_descent_runner(
    points, starting_a, starting_b, learning_rate, num_iterations
):

    a = starting_a
    b = starting_b

    for i in range(num_iterations):
        a, b = step_gradient(a, b, np.array(points), learning_rate)

    return [a, b]


def step_gradient(a_current, b_current, points, learning_rate):

    # starting points for gradient
    a_gradient = 0
    b_gradient = 0

    N = float(len(points))

    for i in range(0, len(points)):
        math = points[i, 0]
        read = points[i, 1]

        # direction for a and b
        a_gradient += -(2 / N) * math * (read - (a_current * math + b_current))
        b_gradient += -(2 / N) * (read - (a_current * math + b_current))

    new_a = a_current - (learning_rate * a_gradient)
    new_b = b_current - (learning_rate * b_gradient)
    if debug:
        print(
            f"Iteration: a={a_current}, b={b_current}, gradients: a_grad={a_gradient}, b_grad={b_gradient}"
        )
    return new_a, new_b


debug = False
learning_rate = 0.0001
initial_a = 0
initial_b = 0
num_iterations = 1000


def main():
    # Define hyperparams
    learning_rate = 0.0001
    initial_a = 0
    initial_b = 0
    num_iterations = 1000
    path = "Cleaned_Students_Performance.csv"

    points = get_points(path)

    print(
        f"Starting point at: a = {initial_a}, b = {initial_b} error = {compute_error(initial_a, initial_b, points)}"
    )
    [a, b] = gradient_descent_runner(
        points, initial_a, initial_b, learning_rate, num_iterations
    )
    print(f"Ending point at: a = {a}, b = {b}, error = {compute_error(a, b, points)}")

    draw_plot(points[:, 0], points[:, 1], a, b)


main()
