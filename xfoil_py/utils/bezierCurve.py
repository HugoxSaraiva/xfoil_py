from scipy.special import comb
import numpy as np
import matplotlib.pyplot as plt


def bezier_curve_2d(control_points):
    """
    Returns a function that calculates the x,y values of the corresponding bezier curve given the control points
    :param control_points: list of control points
    :return: function
    """
    n = len(control_points) - 1
    control_points_array = np.array(control_points)
    return lambda t: sum(
        comb(n, i) * (1 - t) ** (n - i) * t ** i * p for i, p in enumerate(control_points_array)
    )


def bezier_curve_2d_derivative(control_points):
    n = len(control_points) - 1
    control_points_array = np.array(control_points)

    def curve(t):
        result = 0
        for i in range(n):
            result += n*comb(n-1, i) * (1-t)**(n-1-i)*t ** i * (np.subtract(control_points[i+1], control_points[i]))
        return result

    return curve


if __name__ == '__main__':
    curve = bezier_curve_2d([(0, 0), (4, 4), (4, 0), (0, 2)])
    derivative = bezier_curve_2d_derivative([(0, 0), (4, 4), (4, 0), (0, 2)])
    P = []
    for x in np.linspace(0, 1, 100):
        P.append(curve(x))
    t = np.linspace(0, 1, 100)
    plt.plot(*zip(*P))
    plt.show()
    threshold = 0.01
    t1, t2 = 0, 1
    i = 1
    d = 1
    while d > threshold:
        dx, dy = [p for p in curve(t1) - curve(t2)]
        d = (dx ** 2 + dy ** 2) ** (1/2)
        t1, t2 = t1-0.5*d/derivative(t1), t2-0.5*d/derivative(t2)
        print("Distance: {}".format(d))

    # D = []
    # for t in np.linspace(0, 1, 10):
    #     D.append(derivative(t))
    # print(np.shape(D[0][:]))
    # plt.figure()
    # plt.plot(np.linspace(0, 1, 10), [x for x, y in D])
    # plt.plot(np.linspace(0, 1, 10), [y for x, y in D])
    # plt.show()
