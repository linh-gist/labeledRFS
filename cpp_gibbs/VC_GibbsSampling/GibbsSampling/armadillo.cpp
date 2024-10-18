/*
Your Armadillo path could be, C:\armadillo.

In the Visual Studio solution, do the following:

Add armadillo directory under Property Manager --> C/C++ --> General --> Additional Include Directories, add semicolon after existing entries, followed by C:\armadillo\include;%(AdditionalIncludeDirectories)
If you are using 64-bit version to build also do the following:

Property Manager --> Linker --> General --> Additional Library Directories, add semicolon after existing entries, followed by C:\armadillo\examples\lib_win64;%(AdditionalLibraryDirectories)
Property Manager --> Linker --> Input --> Additional Dependencies, add semicolon after existing entries, followed by libopenblas.lib; blas_win64_MT.lib;lapack_win64_MT.lib;%(AdditionalDependencies)
Ensure that you are modifying the Property Manager in the same Solution Configuration and Solution Platform that you are using for the build. If this was successful, you should be able to use armadillo by simply adding #include <armadillo> in your header file.
*/

#include <armadillo>

using namespace arma;

template <typename T>
inline bool approx_equal_cpp(const T& lhs, const T& rhs, double tol = 0.00000001) {
	return arma::approx_equal(lhs, rhs, "absdiff", tol);
}

arma::mat unique_rows(const arma::mat& m) {
	arma::uvec ulmt = arma::zeros<arma::uvec>(m.n_rows);
	for (arma::uword i = 0; i < m.n_rows; i++) {
		for (arma::uword j = i + 1; j < m.n_rows; j++) {
			if (approx_equal_cpp(m.row(i), m.row(j))) { ulmt(j) = 1; break; }
		}
	}
	return m.rows(find(ulmt == 0));
}

arma::mat gibbs_jointpredupdt(arma::mat P0, int m) {
	size_t p0_row = arma::size(P0)[0];
	size_t p0_col = arma::size(P0)[1];
	arma::rowvec currsoln(p0_row);

	arma::mat assignments(m, p0_row, fill::zeros);

	for (int i = 0; i < p0_row; i++)
		currsoln(i) = p0_row + i;

	uvec indices(1); indices(0) = 0;
	assignments.each_row(indices) = currsoln;

	int rand_idx = 0;
	arma::vec rand = arma::vec((m - 1)*p0_row, arma::fill::randu);
	arma::vec tempsamp(p0_col);
	arma::vec idxold(p0_col+1, fill::zeros);
	arma::vec cumsum(p0_col+1, fill::zeros);
	for (int sol = 1; sol < m; sol++) {
		for (int var = 0; var < p0_row; var++) {
			// grab row of costs for current association variable
			for (int i = 0; i < p0_col; i++) {
				tempsamp(i) = exp(-P0(var, i));
			}
			// lock out current and previous iteration step assignments except for the one in question
			for (int i = 0; i < p0_row && i != var; i++) {
				tempsamp(currsoln(i)) = 0.0;
			}

			int idx = 1;
			for (int i = 0; i < p0_col; i++) {
				if (tempsamp(i) > 0) {
					idxold(idx) = i;
					cumsum(idx) = cumsum(idx - 1) + tempsamp(i);
					idx++;
				}
			}
			double sum_tempsamp = cumsum(idx - 1);
			for (int i = 0; i < idx; i++) {
				if ((cumsum(i) / sum_tempsamp) > rand(rand_idx)) {
					currsoln(var) = i;
					break;
				}
			}
			rand_idx++;
			currsoln(var) = idxold(currsoln(var));
		}
		indices(0) = sol;
		assignments.each_row(indices) = currsoln;
	}
	arma::mat result = unique_rows(assignments);
	return result;
}

/*#include<iostream>
#include "pch.h"
#include "test_matrix.h"
#include <chrono>
#include<vector>
using namespace std::chrono;

arma::mat std2mat(vector<vector<double> > A) {
	arma:mat mat_A(A.size(), A[0].size());
	for (size_t i = 0; i < A.size(); ++i) {
		for (size_t j = 0; j < A[0].size(); ++j) {
			mat_A(i,j) = A[i][j];
		}
	};
	return mat_A;
}

int mainaaa() {
	auto matrix = get_matrix();
	mat matrix_arma = std2mat(matrix);
	int test_num = 1000;
	mat result;

	auto start = high_resolution_clock::now();
	for (int i = 0; i < test_num; i++) {
		result = gibbs_jointpredupdt(matrix_arma, 1000);
	}
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);

	cout << "Time taken by function: "
		<< duration.count() / test_num / pow(10, 6) << " seconds" << endl;

	for (unsigned int i = 0; i < arma::size(result)[0]; ++i) {
		for (unsigned int j = 0; j < arma::size(result)[1]; ++j) {
			cout << result(i, j) << " ";
		}
		cout << "\n";
	}

	return 0;
}*/