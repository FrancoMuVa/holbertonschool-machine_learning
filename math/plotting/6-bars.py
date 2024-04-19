#!/usr/bin/env python3
"""
    Function that plot a stacked bar graph.
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """ plot a stacked bar graph """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4,3))
    plt.figure(figsize=(6.4, 4.8))

    color = ['red', 'yellow', 'orange', '#ffe5b4']
    fruits = ['apples', 'bananas', 'oranges', 'peaches']
    person = ['Farrah', 'Fred', 'Felicia']

    bottom = np.zeros(len(person))

    for i in range(len(fruits)):
        plt.bar(person, fruit[i], label=fruits[i], color=color[i], bottom=bottom)
        bottom += fruit[i]

    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.ylim(0, 80)
    plt.legend(loc='upper right')
    plt.show()
