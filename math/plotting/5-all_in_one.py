#!/usr/bin/env python3
"""
    Function that plot all 5 previous graphs in one figure
"""
import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    " Plot all 5 previous graphs in one figure "
    fig, axs = plt.subplots(3, 2, figsize=(6.4, 5.5))
    fig.suptitle('All in One')

    # Line graph
    y0 = np.arange(0, 11) ** 3
    axs[0, 0].plot(y0, 'r-')
    axs[0, 0].set_xlim(0, 10)

    # Scatter graph
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180
    axs[0, 1].plot(x1, y1, 'mo')
    axs[0, 1].set_title('Men\'s Height vs Weight', fontsize='x-small')
    axs[0, 1].set_xlabel('Height (in)', fontsize='x-small')
    axs[0, 1].set_ylabel('Weight (lbs)', fontsize='x-small')

    # Exponential Decay
    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)
    axs[1, 0].plot(x2, y2)
    axs[1, 0].set_title('Exponential Decay of C-14', fontsize='x-small')
    axs[1, 0].set_xlabel('Time (years)', fontsize='x-small')
    axs[1, 0].set_ylabel('Fraction Remaining', fontsize='x-small')
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_xlim(0, 28650)

    # Two line graphs
    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)
    axs[1, 1].plot(x3, y31, 'r--', label='C-14')
    axs[1, 1].plot(x3, y32, 'g-', label='Ra-226')
    axs[1, 1].set_xlim(0, 20000)
    axs[1, 1].set_ylim(0, 1)
    axs[1, 1].set_title('Exponential Decay of Radioactive Elements',
                        fontsize='x-small')
    axs[1, 1].set_xlabel('Time (years)', fontsize='x-small')
    axs[1, 1].set_ylabel('Fraction Remaining', fontsize='x-small')
    axs[1, 1].legend(loc=1, fontsize='x-small')

    # Frequency
    fig.delaxes(axs[2, 1])
    fig.delaxes(axs[2, 0])
    ax = fig.add_subplot(3, 1, 3)
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    m = max(student_grades)
    bin = np.arange(0, 101, 10)
    ax.hist(student_grades, bins=bin, edgecolor='black')
    ax.set_title('Project A', fontsize='x-small')
    ax.set_xlabel('Grades', fontsize='x-small')
    ax.set_ylabel('Number of Students', fontsize='x-small')
    ax.set_xticks(np.arange(0, m, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)

    plt.tight_layout()
    plt.savefig('all_in_one.png')
    plt.show()
