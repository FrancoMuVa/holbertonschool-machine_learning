#!/usr/bin/env python3
"""
    Function that plot a stacked bar graph.
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    " Plot a stacked bar graph "
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))
    color = ['red', 'yellow', '#ff8000', '#ffe5b4']
    fruits_names = ['apples', 'bananas', 'oranges', 'peaches']
    person = ['Farrah', 'Fred', 'Felicia']
    bottom = np.zeros(len(person))
    for i in range(len(fruits_names)):
        plt.bar(person, fruit[i], width=0.5, color=color[i],
                bottom=bottom, label=fruits_names[i])
        bottom += fruit[i]
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.legend(loc=1)
    plt.show()
