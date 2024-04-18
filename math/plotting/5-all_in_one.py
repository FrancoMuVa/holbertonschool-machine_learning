#!/usr/bin/env python3
"""
    Function that plot all 5 previous graphs in one figure.
"""
import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """ plot all 5 previous graphs in one figure """
    fig, axs = plt.subplots(3, 2, figsize=(6.4, 5.5))
    fig.suptitle('All in One')

    # Line Graph
    y0 = np.arange(0, 11) ** 3
    axs[0, 0].plot(y0, 'r-')
    axs[0, 0].set_xlim(0, 10)

    # Scatter
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180
    axs[0, 1].plot(x1, y1, 'mo')
    axs[0, 1].set_title("Men's Height vs Weight", fontsize='x-small')
    axs[0, 1].set_xlabel('Height (in)', fontsize='x-small')
    axs[0, 1].set_ylabel('Weight (lbs)', fontsize='x-small')

    # Change of scale
    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)
    axs[1, 0].plot(x2, y2)
    axs[1, 0].set_title('Exponential Decay of C-14', fontsize='x-small')
    axs[1, 0].set_xlabel('Time (years)', fontsize='x-small')
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_ylabel('Fraction Remaining', fontsize='x-small')
    axs[1, 0].set_xlim(0, 28650)

    # Two is better than one
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
    axs[1, 1].legend(loc='upper right', fontsize='x-small')

    # Frequency
    fig.delaxes(axs[2, 1])
    fig.delaxes(axs[2, 0])
    ax_combined = fig.add_subplot(3, 1, 3)
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    max_range = max(student_grades)
    ax_combined.hist(student_grades, bins=np.arange(0, max_range, 10),
                     range=(0, max(student_grades)),
                     edgecolor='black')
    ax_combined.set_title('Project A', fontsize='x-small')
    ax_combined.set_xlabel('Grades', fontsize='x-small')
    ax_combined.set_ylabel('Number of Students', fontsize='x-small')

    ax_combined.set_xticks(np.arange(0, max_range, 10))
    ax_combined.set_yticks(np.arange(0, max_range, 10))
    ax_combined.set_ylim(0, 30)
    ax_combined.set_xlim(0, 100)
    plt.tight_layout()
    plt.show()
