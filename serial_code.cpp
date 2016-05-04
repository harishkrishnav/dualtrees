#include <iostream>
#include <string>
#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <armadillo>

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
	std::cout << reference_data.n_cols << " " <<  reference_data.n_rows<< std::endl;
	std::cout << query_data.n_cols << " " << reference_data.n_rows << std::endl;
	AllKNN a(reference_data);
	arma::Mat<size_t> resulting_neighbors;
	arma::mat resulting_distances;
	a.Search(query_data, 1, resulting_neighbors, resulting_distances);
	/*for (size_t i = 0; i < resulting_neighbors.n_elem; ++i) {  
		std::cout << "pt : " << i << " neigbors : " << resulting_neighbors[i] << " distance : "<< resulting_distances[i] << "\n";
	}*/
}
