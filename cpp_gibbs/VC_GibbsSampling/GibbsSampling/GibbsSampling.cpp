// GibbsSampling.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <vector>
#include <math.h>       /* exp */
#include <random>		/*uniform distribution*/
#include <tuple>		/* return assignments & costs*/

using namespace  std;

std::tuple<vector<vector<int>>, vector<double>> gibbs_jointpredupdt(vector<vector<double>> P0, int m) {
	std::random_device rand_dev;
	std::mt19937 generator(rand_dev());
	uniform_real_distribution<double> distribution(0.0, 1.0);

	size_t p0_row = P0.size();
	size_t p0_col = P0[0].size();

	if (m == 0) {
		m = 1;
	}

	vector<vector<int>> assignments(m, vector<int>(p0_row));
	vector<int> currsoln(p0_row);
	vector<double> tempsamp(p0_col);
	vector<int> idxold(p0_col + 1, 0);
	vector<double> cumsum(p0_col + 1, 0.0);

	// compute exp of P0 out of 'sol' loop to save computation
	vector<vector<double>> P0_exp(p0_row, vector<double>(p0_col));
	for (int i = 0; i < p0_row; i++) {
		for (int j = 0; j < p0_col; j++) {
			P0_exp[i][j] = exp(-P0[i][j]);
		}
		currsoln[i] = p0_row + i;
	}
	// use all missed detections as initial solution
	assignments[0] = currsoln;

	for (int sol = 1; sol < m; sol++) {
		for (int var = 0; var < p0_row; var++) {
			// grab row of costs for current association variable
			tempsamp = P0_exp[var];
			// lock out current and previous iteration step assignments except for the one in question
			for (int i = 0; i < p0_row; i++) {
				if (i == var)
					continue;
				tempsamp[currsoln[i]] = 0.0;
			}
			int idx = 1;
			for (int i = 0; i < p0_col; i++) {
				if (tempsamp[i] > 0) {
					idxold[idx] = i;
					cumsum[idx] = cumsum[idx - 1] + tempsamp[i];
					idx++;
				}
			}
			double sum_tempsamp = cumsum[idx - 1];
			double rand_num = distribution(generator);
			for (int i = 1; i < idx; i++) {
				if ((cumsum[i] / sum_tempsamp) > rand_num) {
					currsoln[var] = i;
					break;
				}
			}
			currsoln[var] = idxold[currsoln[var]];
		}
		assignments[sol] = currsoln;
	}
	std::sort(assignments.begin(), assignments.end());
	assignments.erase(std::unique(assignments.begin(), assignments.end()), assignments.end());

	// calculate costs for each assignment
	int costs_size = assignments.size();
	vector<double> costs(costs_size);
	for (int i = 0; i < costs_size; i++) {
		for (int j = 0; j < p0_row; j++) {
			costs[i] += P0[j][assignments[i][j]];
		}
	}

	return std::make_tuple(assignments, costs);
}

#include<iostream>
#include "pch.h"
#include "test_matrix.h"
#include <chrono>
using namespace std::chrono;

void main()
{
	vector<vector<int>> assignments;
	vector<double> cost;
	auto matrix = get_matrix();
	int test_num = 1000;

	auto start = high_resolution_clock::now();
	for (int i = 0; i < test_num; i++) {
		std::tie(assignments, cost) = gibbs_jointpredupdt(matrix, 1000);
	}
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);

	cout << "Time taken by functionaaa: "
		<< duration.count() / test_num / pow(10, 6) << " seconds" << endl;

	for (int i = 0; i < assignments.size(); i++) {
		for (int j = 0; j < assignments[0].size(); j++) {
			cout << assignments[i][j] << " ";
		}
		cout << cost[i] << "\n";
	}
}