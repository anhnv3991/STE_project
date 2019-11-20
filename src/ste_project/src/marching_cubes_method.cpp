#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/surface/marching_cubes_rbf.h>
#include <pcl/filters/voxel_grid.h>

#include <sstream>

int main (int argc, char** argv)
{
	if (argc != 4) {
		std::cout << "Usage: Command <input_file> <output_dir> <resolution>" << std::endl;
		return -1;
	}

	float resolution = std::stof(argv[3]);

	pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud (new pcl::PointCloud<pcl::PointXYZ>);

	pcl::io::loadPLYFile<pcl::PointXYZ>(argv[1], *in_cloud);

	// Downsampling
	// Step 1.5: Downsampling in_cloud
	pcl::PointCloud<pcl::PointXYZ>::Ptr down_sampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);

	down_sampled_cloud = in_cloud;
//	pcl::VoxelGrid<pcl::PointXYZ> ds_filter;
//
//	ds_filter.setInputCloud(in_cloud);
//	ds_filter.setLeafSize(resolution, resolution, resolution);
//	ds_filter.filter(*down_sampled_cloud);
//
//	std::cout << "Number of downsampled points = " << down_sampled_cloud->size() << std::endl;

	// Estimate normals
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree1 (new pcl::search::KdTree<pcl::PointXYZ>);

    tree1->setInputCloud (down_sampled_cloud);

    ne.setInputCloud (down_sampled_cloud);
    ne.setSearchMethod (tree1);
    ne.setKSearch (20);

    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);

    ne.compute (*normals);

    // Concatenate the XYZ and normal fields*
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields(*down_sampled_cloud, *normals, *cloud_with_normals);

    // Create search tree*
    pcl::search::KdTree<pcl::PointNormal>::Ptr tree (new pcl::search::KdTree<pcl::PointNormal>);

    tree->setInputCloud (cloud_with_normals);

    std::cout << "begin marching cubes reconstruction" << std::endl;

    pcl::MarchingCubesRBF<pcl::PointNormal> mc;
    pcl::PolygonMesh::Ptr triangles(new pcl::PolygonMesh);

    mc.setInputCloud (cloud_with_normals);
    mc.setSearchMethod (tree);
    mc.reconstruct (*triangles);

    std::cout << triangles->polygons.size() << " triangles created" << std::endl;

    // Save to vtk file
    std::stringstream ss;

    ss << argv[2] << "marching_cubes.vtk";

    pcl::io::saveVTKFile(ss.str(), *triangles);

    return 0;
}
