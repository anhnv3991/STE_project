#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/don.h>
#include <pcl/filters/voxel_grid.h>

#include <string>
#include <sstream>


int main(int argc, char *argv[])
{
	if (argc != 3) {
		std::cout << "Usage: command <in_ply_file> <out_obj_file> <scale1> <scale2> <threshold> <segradius>" << std::endl;
		exit(EXIT_FAILURE);
	}

	// Step 0: read command parameters
	// scales and threshold used for Difference of Normals (DoN) filter
	double scale1 = 0.03;
	double scale2 = 0.04;
	double threshold = 0.03;
	double segradius = 0.04;

//	std::istringstream(argv[3]) >> scale1;
//	std::istringstream(argv[4]) >> scale2;
//	std::istringstream(argv[5]) >> threshold;
//	std::istringstream(argv[6]) >> segradius;

	// Input and output file names
	std::string in_ply_fname(argv[1]);
	std::string out_obj_fname(argv[2]);

	// Step 1: Load data from input ply file to a point cloud
	pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPLYFile(in_ply_fname, *in_cloud);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

	// Tes the imported point cloud
	pcl::io::savePLYFile("loaded_cloud.ply", *in_cloud);

	// Step 1.5: Downsampling
	pcl::VoxelGrid<pcl::PointXYZ> sor;

	sor.setInputCloud(in_cloud);
	sor.setLeafSize(0.005, 0.005, 0.005);
	sor.filter(*cloud);

	// Save to a file to check
	pcl::io::savePLYFile("down_sampled_cloud.ply", *cloud);

	// Step 2: Estimate normals
	pcl::search::Search<pcl::PointXYZ>::Ptr tree;

	if (cloud->isOrganized()) {
		tree.reset(new pcl::search::OrganizedNeighbor<pcl::PointXYZ>());
	} else {
		tree.reset(new pcl::search::KdTree<pcl::PointXYZ>(false));
	}

	tree->setInputCloud(cloud);

	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::PointNormal> ne;

	ne.setInputCloud(cloud);
	ne.setSearchMethod(tree);

	// View point = infinity
	ne.setViewPoint(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());

	// Calculate normals with the smalle scale
	std::cout << "Calculating normals for small scale..." << scale1 << std::endl;
	pcl::PointCloud<pcl::PointNormal>::Ptr normals_small_scale(new pcl::PointCloud<pcl::PointNormal>);

	ne.setRadiusSearch(scale1);
	ne.compute(*normals_small_scale);

	// Calculate normals with the large scale
	std::cout << "Calculating normals for large scale..." << scale2 << std::endl;
	pcl::PointCloud<pcl::PointNormal>::Ptr normals_large_scale(new pcl::PointCloud<pcl::PointNormal>);

	ne.setRadiusSearch(scale2);
	ne.compute(*normals_large_scale);

	// Output cloud for DoN results
	pcl::PointCloud<pcl::PointNormal>::Ptr don_cloud(new pcl::PointCloud<pcl::PointNormal>);
	pcl::copyPointCloud(*cloud, *don_cloud);

	// Create DoN operator
	pcl::DifferenceOfNormalsEstimation<pcl::PointXYZ, pcl::PointNormal, pcl::PointNormal> don;
	don.setInputCloud(cloud);
	don.setNormalScaleLarge(normals_large_scale);
	don.setNormalScaleSmall(normals_small_scale);

	if (!don.initCompute()) {
		std::cerr << "Error: Could not initialize DoN feature operator" << std::endl;
		exit(EXIT_FAILURE);
	}

	don.computeFeature(*don_cloud);

	pcl::io::savePLYFile("computed_feature_don_cloud.ply", *don_cloud);

	std::cout << "Filtering out DoN magnitude <= " << threshold << std::endl;

	// Build condition for filtering
	pcl::ConditionOr<pcl::PointNormal>::Ptr range_cond(new pcl::ConditionOr<pcl::PointNormal>());

	range_cond->addComparison(pcl::FieldComparison<pcl::PointNormal>::ConstPtr(
								new pcl::FieldComparison<pcl::PointNormal>("curvature", pcl::ComparisonOps::GT, threshold)));

	// Build the filter
	pcl::ConditionalRemoval<pcl::PointNormal> condrem;
	condrem.setCondition(range_cond);
	condrem.setInputCloud(don_cloud);

	pcl::PointCloud<pcl::PointNormal>::Ptr don_cloud_filtered(new pcl::PointCloud<pcl::PointNormal>);

	condrem.filter(*don_cloud_filtered);

	don_cloud = don_cloud_filtered;

	pcl::io::savePLYFile("cond_filtered_don_cloud.ply", *don_cloud);

	std::cout << "Clustering using EuclideanClusterExtraction with tolerance <= " << segradius << std::endl;

	pcl::search::KdTree<pcl::PointNormal>::Ptr segtree(new pcl::search::KdTree<pcl::PointNormal>);
	segtree->setInputCloud(don_cloud);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointNormal> ec;

	ec.setClusterTolerance(segradius);
	ec.setMinClusterSize(50);
	ec.setMaxClusterSize(100000);
	ec.setSearchMethod(segtree);
	ec.setInputCloud(don_cloud);
	ec.extract(cluster_indices);

	int j = 0;

	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it, ++j) {
		pcl::PointCloud<pcl::PointNormal>::Ptr cloud_cluster_don(new pcl::PointCloud<pcl::PointNormal>);

		for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) {
			cloud_cluster_don->points.push_back(don_cloud->points[*pit]);
		}

		cloud_cluster_don->width = int(cloud_cluster_don->points.size());
		cloud_cluster_don->height = 1;
		cloud_cluster_don->is_dense = true;

		// Save cluster
		std::stringstream ss;
		ss << "don_cluster_" << j << ".ply";
		pcl::io::savePLYFile(ss.str(), *cloud_cluster_don);
	}

	return 0;
}
