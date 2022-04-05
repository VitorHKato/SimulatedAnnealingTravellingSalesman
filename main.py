import numpy
import cv2
import time

WIDTH = 800
HEIGHT = 600
STOPPING_TEMPERATURE = 1
INITIAL_TEMPERATURE = 1000
FONT = cv2.FONT_HERSHEY_COMPLEX
SIZE = 0.7
WHITE = (255, 255, 255)
RED = (0, 0, 255)


# Generate random in (x, y) positions
def create_cities(width, height, count):
    cities = []
    for i in range(count):
        position_x = numpy.random.randint(width)
        position_y = numpy.random.randint(height)
        cities.append((position_x, position_y))
    return cities

    # Fixed to compare
    # return [(779, 74), (403, 343), (551, 395), (525, 519), (346, 526), (604, 264), (591, 228)]


# Choose a random path (solution)
def get_path(count):
    solution = numpy.arange(count)
    numpy.random.shuffle(solution)
    return solution


'''
    :param cities: array of cities positions
    :param solution: current sorted solution path
    :return: total distance traveled
    
'''
def get_total_distance(cities, solution):
    distance = 0
    for i in range(len(cities)):
        index_a = solution[i]
        index_b = solution[i - 1]
        delta_x = cities[index_a][0] - cities[index_b][0]
        delta_y = cities[index_a][1] - cities[index_b][1]
        distance += (delta_x ** 2 + delta_y ** 2) ** 0.5
    return distance


# Generate another path changing 2 positions
def get_another_solution(current):
    new = current.copy()
    index_a = numpy.random.randint(len(current))
    index_b = numpy.random.randint(len(current))
    while index_b == index_a:
        index_b = numpy.random.randint(len(current))
    new[index_a], new[index_b] = new[index_b], new[index_a]
    return new


def draw(width, height, cities, solution, infos):
    frame = numpy.zeros((height, width, 3))
    for i in range(len(cities)):
        index_a = solution[i]
        index_b = solution[i - 1]
        point_a = (cities[index_a][0], cities[index_a][1])
        point_b = (cities[index_b][0], cities[index_b][1])
        cv2.line(frame, point_a, point_b, WHITE, 2)
    for city in cities:
        cv2.circle(frame, (city[0], city[1]), 5, RED, -1)
    cv2.putText(frame, f"Temperature", (25, 50), FONT, SIZE, WHITE)
    cv2.putText(frame, f"Score", (25, 75), FONT, SIZE, WHITE)
    cv2.putText(frame, f"Best Score", (25, 100), FONT, SIZE, WHITE)
    cv2.putText(frame, f"Worst Score", (25, 125), FONT, SIZE, WHITE)
    cv2.putText(frame, f": {infos[0]:.2f}", (175, 50), FONT, SIZE, WHITE)
    cv2.putText(frame, f": {infos[1]:.2f}", (175, 75), FONT, SIZE, WHITE)
    cv2.putText(frame, f": {infos[2]:.2f}", (175, 100), FONT, SIZE, WHITE)
    cv2.putText(frame, f": {infos[3]:.2f}", (175, 125), FONT, SIZE, WHITE)
    cv2.imshow("Simulated Annealing Traveling Salesman", frame)
    cv2.waitKey(5)


if __name__ == "__main__":
    num_cities = int(input("Enter the number of cities: "))

    temperature_decrease = 0.995

    cities = create_cities(WIDTH, HEIGHT, num_cities)

    t1 = time.time()

    current_solution = get_path(num_cities)
    current_score = get_total_distance(cities, current_solution)

    print("Initial score: {}".format(current_score))

    best_score = worst_score = current_score
    actual_temperature = INITIAL_TEMPERATURE

    while (actual_temperature > STOPPING_TEMPERATURE):
        new_solution = get_another_solution(current_solution)
        new_score = get_total_distance(cities, new_solution)
        best_score = min(best_score, new_score)
        worst_score = max(worst_score, new_score)

        if new_score <= current_score:
            current_solution = new_solution
            current_score = new_score
        else:
            # If the new score is bad, we will just ignore the current path and score.
            # The probability of acceptance decreases with the temperature as well

            delta = new_score - current_score
            probability = numpy.exp(-delta / actual_temperature)
            if probability > numpy.random.uniform():
                current_solution = new_solution
                current_score = new_score

        actual_temperature *= temperature_decrease
        infos = (actual_temperature, current_score, best_score, worst_score)
        draw(WIDTH, HEIGHT, cities, current_solution, infos)

    t2 = time.time()

    total_time = t2 - t1

    print("Best Score: {}".format(current_score))
    print("Best solution: {}".format(current_solution))
    print("Total time: {}s".format(total_time))