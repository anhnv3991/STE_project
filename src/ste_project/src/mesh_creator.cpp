#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/don.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/gp3.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/obj_io.h>

#include <stdlib.h>
#include <vector>

int main(int argc, char **argv)
{
	if (argc != 7) {
		std::cout << "Usage: rosrun ste_project mesh_creator <in_ply_file> <out_folder> <resolution> <scale1> <scale2> <segradius>" << std::endl;
		exit(EXIT_FAILURE);
	}

	std::string in_ply_name(argv[1]);
	std::string out_dir_name(argv[2]);

	double resolution = std::atof(argv[3]);
	double small_scale = std::atof(argv[4]);
	double large_scale = std::atof(argv[5]);
	double segradius = std::atof(argv[6]);

	pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>);

	// Step 1: Load from ply file to in_cloud
	pcl::io::loadPLYFile(in_ply_name, *in_cloud);

	// Step 1.5: Downsampling in_cloud
	pcl::PointCloud<pcl::PointXYZ>::Ptr down_sampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::VoxelGrid<pcl::PointXYZ> ds_filter;

	ds_filter.setInputCloud(in_cloud);
	ds_filter.setLeafSize(resolution, resolution, resolution);
	ds_filter.filter(*down_sampled_cloud);

	// Step 2: Estimate normals
	pcl::search::Search<pcl::PointXYZ>::Ptr tree;

	if (in_cloud->isOrganized()) {
		tree.reset(new pcl::search::OrganizedNeighbor<pcl::PointXYZ>());
	} else {
		tree.reset(new pcl::search::KdTree<pcl::PointXYZ>(false));
	}

	tree->setInputCloud(down_sampled_cloud);

	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
	pcl::PointCloud<pcl::Normal>::Ptr normal_small_scale(new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<pcl::Normal>::Ptr normal_large_scale(new pcl::PointCloud<pcl::Normal>);

	ne.setInputCloud(down_sampled_cloud);
	ne.setSearchMethod(tree);;
	ne.setViewPoint(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());

	// Compute small scale normals
	ne.setRadiusSearch(small_scale);
	ne.compute(*normal_small_scale);

	// Compute large scale normals
	ne.setRadiusSearch(large_scale);
	ne.compute(*normal_large_scale);

	// Compute DoN
	pcl::PointCloud<pcl::PointNormal>::Ptr don_cloud(new pcl::PointCloud<pcl::PointNormal>);
	pcl::copyPointCloud(*down_sampled_cloud, *don_cloud);

	pcl::DifferenceOfNormalsEstimation<pcl::PointXYZ, pcl::Normal, pcl::PointNormal> don;

	don.setInputCloud(down_sampled_cloud);
	don.setNormalScaleLarge(normal_large_scale);
	don.setNormalScaleSmall(normal_small_scale);

	if (!don.initCompute()) {
		std::cerr << "Error: Could not initialize DoN feature operator!" << std::endl;
		exit(EXIT_FAILURE);
	}

	don.computeFeature(*don_cloud);

	// Segmentation
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
		pcl::PointCloud<pcl::Normal>::Ptr normal_cluster(new pcl::PointCloud<pcl::Normal>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normal(new pcl::PointCloud<pcl::PointNormal>);

		for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) {
			normal_cluster->points.push_back(normal_small_scale->points[*pit]);
			cloud_cluster->points.push_back(down_sampled_cloud->points[*pit]);
		}

		pcl::concatenateFields(*cloud_cluster, *normal_cluster, *cloud_with_normal);
		// Triangulation
		// Create search tree*
		pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);
		tree2->setInputCloud (cloud_with_normal);

		// Initialize objects
		pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
		pcl::PolygonMesh triangles;

		// Set the maximum distance between connected points (maximum edge length)
		gp3.setSearchRadius (0.1);

		// Set typical values for the parameters
		gp3.setMu (2.5);
		gp3.setMaximumNearestNeighbors (100);
		gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
		gp3.setMinimumAngle(M_PI/18); // 10 degrees
		gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
		gp3.setNormalConsistency(false);

		// Get result
		gp3.setInputCloud (cloud_with_normal);
		gp3.setSearchMethod (tree2);
		gp3.reconstruct (triangles);

		std::stringstream ss;

		std::cout << "Triangle num = " << triangles.polygons.size() << std::endl;
		ss << out_dir_name << "mesh_" << j << ".vtk";

		pcl::io::saveVTKFile(ss.str(), triangles);

	}



	return 0;
}
