#include <iostream>
#include <string>
#include <vector>
#include <thread>

#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <armadillo>

using namespace mlpack::neighbor;
typedef NeighborSearch<NearestNeighborSort, 
		mlpack::metric::EuclideanDistance,
		arma::mat,
		mlpack::tree::KDTree> AllKNN;

// global variable for the subtree answers
std::vector<arma::Mat<size_t> *> subtree_neighbors;
std::vector<arma::mat *> subtree_distances;

// The worker function to find the nearest neighbour 
// in the subtrees for the whole query tree. 
// This basically is the mapper function 
// which will run in parallel.
void do_NN_subtree(arma::Mat<size_t> *subtree_pts, arma::mat *subtree_dist, 
			const arma::mat& ref_subtree, size_t k,
			const arma::mat& query_data) {
	
	AllKNN a(ref_subtree);
	arma::Mat<size_t> resulting_neighbors;
	arma::mat resulting_distances;
	a.Search(query_data, k, resulting_neighbors, resulting_distances);
	*subtree_pts = arma::Mat<size_t>(resulting_neighbors);
	*subtree_dist = arma::mat(resulting_distances);
}

int main() {
	size_t k = 1;
	size_t num_threads = 3;
	arma::mat query_data, reference_data;
	query_data.load("iris_test.csv");
	reference_data.load("iris.csv");
	arma::inplace_trans(query_data);
	arma::inplace_trans(reference_data);
	
	size_t partitions = (size_t)((float)reference_data.n_cols/(float)num_threads);
	/**
	*
	* This is where the submatrices
	* are constructed. 
	*
	**/	
	std::vector<arma::mat> ref_mats;
	arma::mat tmp;
	for(auto j = 0; j < num_threads; j++) {
		tmp = arma::mat(reference_data.submat(0, j*partitions, reference_data.n_rows - 1, (j+1)*partitions - 1));
		ref_mats.push_back(tmp);
		tmp.clear();
	}
	/**
	*
	* This is the mapper portion
	* of the code.
	*
	**/
	// make global pointer arrays	
	for(auto i = 0; i < num_threads; i++) {
		subtree_neighbors.push_back(new arma::Mat<size_t>());
		subtree_distances.push_back(new arma::mat());
	}
	// create worker threads
	std::vector<std::thread *> t;
	for(auto i = 0; i < num_threads; i++) {
		t.push_back(new std::thread(do_NN_subtree, subtree_neighbors[i], subtree_distances[i], ref_mats[i], k, query_data));
	}
	// Wait for all threads to end
	for(auto i = 0; i < num_threads; i++) {
		t[i]->join();
	}
	/**
	*
	* This is the reducer portion
	* of the code.
	*
	**/
	arma::Cube<size_t> global_neighbors((*subtree_neighbors[0]).n_rows,(*subtree_neighbors[0]).n_cols,num_threads);
	arma::cube global_distances((*subtree_neighbors[0]).n_rows,(*subtree_neighbors[0]).n_cols,num_threads);
	for(auto i = 0; i < num_threads; i++) {
		global_neighbors.slice(i) =  *subtree_neighbors[i];
		global_distances.slice(i) =  *subtree_distances[i];
		
	}
	std::cout << global_neighbors << std::endl; 
	for(auto i = 0; i < num_threads; i++) {
		delete t[i];
		delete subtree_neighbors[i];
	}
	return 0;
}

