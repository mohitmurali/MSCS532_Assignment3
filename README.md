# Assignment 3: Algorithm Efficiency and Scalability

## Overview
This repository contains the implementation and analysis of Randomized Quicksort and Hashing with Chaining.

## Summary

This assignment analyzes the efficiency and scalability of Randomized Quicksort and Deterministic Quicksort, comparing their performance on random, sorted, reverse-sorted, and repeated-element arrays of sizes 100, 1000, and 10,000. Empirical results show Randomized Quicksort maintains consistent O(n log n) performance across all array types due to random pivot selection, while Deterministic Quicksort exhibits O(n^2) behavior on sorted and reverse-sorted arrays, with significant time increases as array size grows. For random and repeated arrays, both algorithms perform similarly at smaller sizes, but Deterministic Quicksort scales poorly at larger sizes. The Hash Table with Chaining implementation demonstrates O(1 + α) average-case performance, with load factor management discussed theoretically.

## Files
- `assignment3.py`: Python code for implementations and empirical comparisons.
- `assignment3_report.docx`: APA 7-formatted report with analysis and results.
- `quicksort_*.png`: Plot files generated by `assignment3.py` (e.g., `quicksort_random.png`).

## Running the Code
1. Install dependencies: `pip install matplotlib`.
2. Run: `python3 assignment3.py`.
3. Plots are saved as PNG files in the working directory.

## Report
The report is in `assignment3_report.docx`, formatted in Times New Roman, 12-point, double-spaced, per APA 7 guidelines.
