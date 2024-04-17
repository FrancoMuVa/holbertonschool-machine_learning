#!/usr/bin/env python3
"""
    Function that plot a histogram of student scores for a project.
"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """ plot a histogram of student scores for a project """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    plt.title('Project A')
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')

    max_range = max(student_grades)

    plt.xticks(np.arange(0, max_range, 10))
    plt.yticks(np.arange(0, max_range, 5))
    plt.ylim(0, 30)
    plt.xlim(0, 100)

    plt.hist(student_grades, bins=np.arange(0, max_range, 10),
             range=(0, max(student_grades)),
             edgecolor='black')
    plt.show()
