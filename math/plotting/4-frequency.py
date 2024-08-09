#!/usr/bin/env python3
"""
    Function that plot a histogram of student scores for a project.
"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    " Plot a histogram of student scores for a project. "
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    m = max(student_grades)
    bin = np.arange(0, 101, 10)
    plt.hist(student_grades, bins=bin, edgecolor='black')
    plt.title('Project A')
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.xticks(np.arange(0, m, 10))
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.show()
