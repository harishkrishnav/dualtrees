#include <iostream>
#include <string>
#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <armadillo>

#include <boost/timer/timer.hpp>
#include <boost/chrono/include.hpp>

using namespace mlpack::neighbor;
typedef NeighborSearch<NearestNeighborSort, 
		mlpack::metric::EuclideanDistance,
		arma::mat,
		mlpack::tree::KDTree> AllKNN;

int main() {
	arma::mat query_data, reference_data;
	query_data.load("iris_test.csv");
	reference_data.load("iris.csv");
	arma::inplace_trans(query_data);
	arma::inplace_trans(reference_data);
	
	boost::timer::cpu_timer timer;
	
	AllKNN a(reference_data);
	arma::Mat<size_t> resulting_neighbors;
	arma::mat resulting_distances;
	a.Search(query_data, 1, resulting_neighbors, resulting_distances);
	
	auto nanoseconds = boost::chrono::nanoseconds(timer.elapsed().user + timer.elapsed().system);
	auto seconds = boost::chrono::duration_cast<boost::chrono::seconds>(nanoseconds);
	std::cout << seconds.count() << std::endl;
	
	std::cout << resulting_neighbors << std::endl;
	return 0;
}
