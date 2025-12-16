import numpy as np
import color as color

def zad1():
    print(color.GREEN + "#Zad 1" + color.END)
    chr_size = 10
    pop_size = 10
    generations = 0
    mutation_probability = 0.6

    population = np.random.choice((0, 1), size=(pop_size, chr_size))
    print('Starting population:\n', population)

    while not np.isin(chr_size, np.sum(population, axis=1)):
        sorted_indices = np.argsort(population.sum(axis=1))[::-1]
        population = population[sorted_indices]

        best, sec_best = population[:2]

        cut_index = np.random.choice(len(best))
        kid_1 = np.concatenate((best[:cut_index], sec_best[cut_index:]))
        kid_2 = np.concatenate((sec_best[:cut_index], best[cut_index:]))

        for kid in (kid_1, kid_2):
            if np.random.rand() < mutation_probability:
                index = np.random.randint(chr_size)
                kid[index] = 1 - kid[index]

        population[-2:] = [kid_1, kid_2]

        generations += 1

    print('End population:\n', population)
    print('Generations: ', generations)


def reverse_roulette(array):
    probabilities = 1 / array
    return probabilities / np.sum(probabilities)


def zad2():
    print(color.GREEN + "#Zad 2" + color.END)
    pop_size = 10
    mutation_probability = 0.1
    generations = 0

    population = np.array([
        list(format(np.random.randint(0, 16), "04b") +
             format(np.random.randint(0, 16), "04b"))
        for _ in range(pop_size)
    ], dtype=np.uint8)

    while True:
        list_a = [int("".join(map(str, chrom[:4])), 2) for chrom in population]
        list_b = [int("".join(map(str, chrom[4:])), 2) for chrom in population]

        # Obliczanie różnic (fitness)
        differences = np.abs(33 - (2 * np.power(list_a, 2) + list_b))

        if np.any(differences == 0):
            break

        # Selekcja ruletką
        percentages = reverse_roulette(differences)
        population = population[np.random.choice(range(pop_size), size=pop_size, p=percentages)]

        # Krzyżowanie + mutacja
        for _ in range(len(population) // 2):
            parents_indices = np.random.choice(
                range(len(population)), size=2, replace=False
            )

            cut_index = np.random.choice(range(8))

            parent_1, parent_2 = population[parents_indices]
            kid = np.concatenate((parent_1[:cut_index], parent_2[cut_index:]))

            if np.random.rand() < mutation_probability:
                index = np.random.choice(len(kid))
                kid[index] = 1 - kid[index]

            population[np.random.choice(parents_indices)] = kid

        generations += 1

    print("End population (chromosome, a, b, difference):")
    for chrom in population:
        a = int("".join(map(str, chrom[:4])), 2)
        b = int("".join(map(str, chrom[4:])), 2)
        diff = abs(33 - (2 * (a ** 2) + b))
        print(f"{''.join(map(str, chrom))} -> a={a}, b={b}, difference={diff}")
    print('Generations: ', generations)

# Zad 3

backpack_data = np.array([
    [3, 266],
    [13, 442],
    [10, 671],
    [9, 526],
    [7, 388],
    [1, 245],
    [8, 145],
    [8, 145],
    [2, 126],
    [9, 322],
])


# Funkcja fitness: suma wartości, jeśli waga <= 35, inaczej 0
def calculate_backpack_fitness(population):
    weights = np.sum(population * backpack_data[:, 0], axis=1)
    values = np.sum(population * backpack_data[:, 1], axis=1)
    fitness = np.where(weights <= 35, values, 0)
    return fitness

def zad3():
    print(color.GREEN + "#Zad 3" + color.END)
    mutation_probability = 0.05
    pop_size = 8
    generations = 0

    # losowa populacja: 8 osobników, każdy 10 genów
    population = np.random.choice((0, 1), size=(pop_size, 10))

    while generations < 500:  # limit generacji
        fitness = calculate_backpack_fitness(population)

        # elitarność 25%
        elite_count = pop_size // 4
        elite_indices = np.argsort(fitness)[-elite_count:]
        elite = population[elite_indices]

        # selekcja rulatką
        if np.sum(fitness) == 0:
            percentages = np.ones(pop_size) / pop_size
        else:
            percentages = fitness / np.sum(fitness)

        selected_indices = np.random.choice(range(pop_size), size=pop_size, p=percentages)
        selected = population[selected_indices]

        # krzyżowanie+mutacja
        new_population = list(elite)  # zaczynamy od elity
        while len(new_population) < pop_size:
            parents_indices = np.random.choice(range(pop_size), size=2, replace=False)
            parent_1, parent_2 = selected[parents_indices]

            cut_index = np.random.randint(1, parent_1.shape[0])
            kid = np.concatenate((parent_1[:cut_index], parent_2[cut_index:]))

            # mutacja
            for i in range(len(kid)):
                if np.random.rand() < mutation_probability:
                    kid[i] = 1 - kid[i]

            new_population.append(kid)

        population = np.array(new_population)
        generations += 1

    # wynik
    fitness = calculate_backpack_fitness(population)
    index = np.argmax(fitness)
    best = population[index]
    weight = np.sum(best * backpack_data[:, 0])
    value = np.sum(best * backpack_data[:, 1])

    print(f"Generacje: {generations}")
    print(f"Najlepszy plecak: {best}")
    print(f"Waga: {weight}, Wartość: {value}")
    print(f"Wybrane indeksy: {np.where(best==1)[0]}")

# Zad 4

def calculate_distance(pos_1, pos_2):
    return np.sqrt(np.sum((pos_1 - pos_2) ** 2))


salesman_data = np.array(
    [
        [119, 38],
        [37, 38],
        [197, 55],
        [85, 165],
        [12, 50],
        [100, 53],
        [81, 142],
        [121, 137],
        [85, 145],
        [80, 197],
        [91, 176],
        [106, 55],
        [123, 57],
        [40, 81],
        [78, 125],
        [190, 46],
        [187, 40],
        [37, 107],
        [17, 11],
        [67, 56],
        [78, 133],
        [87, 23],
        [184, 197],
        [111, 12],
        [66, 178],
    ]
)


def calculate_road(cities, city_distances):
    return np.sum(city_distances[cities[:-1], cities[1:]])

def reverse_roulette_roads(roads):
    scores = 1 / (1 + roads)  # krótsze trasy -> większe prawdopodobieństwo
    return scores / np.sum(scores)

def zad4():
    print(color.GREEN + "#Zad 4" + color.END)
    pop_size = 100
    num_cities = len(salesman_data)
    elite_threshold = 0.2
    mutation_probability = 0.01
    generations = 0
    max_generations = 1000

    # macierz odległości
    city_distances = np.array([
        [calculate_distance(outer_city, inner_city) for inner_city in salesman_data]
        for outer_city in salesman_data
    ])

    # populacja początkowa
    population = np.array([
        np.random.choice(range(num_cities), size=num_cities, replace=False)
        for _ in range(pop_size)
    ])

    while generations < max_generations:
        roads = np.array([calculate_road(pop, city_distances) for pop in population])

        # elita
        elite_count = int(pop_size * elite_threshold)
        elite_indices = np.argsort(roads)[:elite_count]
        elite = population[elite_indices]

        # selekcja ruletką
        percentages = reverse_roulette_roads(roads)
        selected_indices = np.random.choice(range(pop_size), size=pop_size, p=percentages)
        selected = population[selected_indices]

        new_population = list(elite)

        # krzyżowanie
        while len(new_population) < pop_size:
            p1, p2 = selected[np.random.choice(range(pop_size), size=2, replace=False)]
            cut1, cut2 = sorted(np.random.choice(range(num_cities), size=2, replace=False))
            kid = np.full(num_cities, -1)
            kid[cut1:cut2] = p1[cut1:cut2]

            fill = [city for city in p2 if city not in kid]
            kid[kid == -1] = fill

            # mutacja
            if np.random.rand() < mutation_probability:
                i, j = np.random.choice(num_cities, size=2, replace=False)
                kid[i], kid[j] = kid[j], kid[i]

            new_population.append(kid)

        population = np.array(new_population)
        generations += 1

    # wynik końcowy
    roads = np.array([calculate_road(pop, city_distances) for pop in population])
    index = np.argmin(roads)
    print(f"Generacje: {generations}")
    print(f"Najlepsza trasa: {population[index]}")
    print(f"Długość trasy: {roads[index]}")

# Generacje: 1000
# Najlepsza trasa: [21 23  0 12 11  5 19  1 18  4 13 17 14 20  6  8 10  9 24  3  7 22  2 15
#  16]
# Długość trasy: 767.9285680511302

zad1()
zad2()
zad3()
zad4()
