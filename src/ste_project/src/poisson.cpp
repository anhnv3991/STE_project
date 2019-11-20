#include <iostream>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/point_types.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/poisson.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>

#include <sstream>

int main (int argc, char* argv[])
{
	if (argc != 4) {
		std::cout << "Usage: Command <input_file> <output_dir> <resolution>" << std::endl;

		return -1;
	}

	std::string input_name(argv[1]);
	float resolution = std::atof(argv[3]);

	pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>);

	pcl::io::loadPLYFile<pcl::PointXYZ>(input_name, *in_cloud);

	pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PassThrough<pcl::PointXYZ> filter;

	filter.setInputCloud(in_cloud);
	filter.filter(*filtered);

	std::cout << "passthrough filter complete" << std::endl;

	// Downsampling

	// Step 1.5: Downsampling in_cloud
	pcl::PointCloud<pcl::PointXYZ>::Ptr down_sampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::VoxelGrid<pcl::PointXYZ> ds_filter;

	ds_filter.setInputCloud(filtered);
	ds_filter.setLeafSize(resolution, resolution, resolution);
	ds_filter.filter(*down_sampled_cloud);

	std::cout << "Number of downsampled points = " << down_sampled_cloud->size() << std::endl;


	std::cout << "begin normal estimation" << std::endl;
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;

	ne.setNumberOfThreads(8);
	ne.setInputCloud(down_sampled_cloud);
	ne.setRadiusSearch(0.05);	// default 0.01
	ne.setViewPoint(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());

	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
	ne.compute(*cloud_normals);
	std::cout << "normal estimation complete" << std::endl;
	std::cout << "reverse normals' direction" << std::endl;

	for(size_t i = 0; i < cloud_normals->size(); ++i) {
		cloud_normals->points[i].normal_x *= -1;
      	cloud_normals->points[i].normal_y *= -1;
      	cloud_normals->points[i].normal_z *= -1;
	}

	std::cout << "combine points and normals" << std::endl;
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_smoothed_normals(new pcl::PointCloud<pcl::PointNormal>);
	pcl::concatenateFields(*down_sampled_cloud, *cloud_normals, *cloud_smoothed_normals);

	std::cout << "begin poisson reconstruction" << std::endl;
	pcl::Poisson<pcl::PointNormal> poisson;

	poisson.setDepth(7);	// default 9
	poisson.setInputCloud(cloud_smoothed_normals);

	pcl::PolygonMesh mesh;

	poisson.reconstruct(mesh);

	std::cout << "Number of polygons = " << mesh.polygons.size() << std::endl;

	std::stringstream ss;


	ss << argv[2] << "possion.vtk";

	pcl::io::saveVTKFile(ss.str(), mesh);

   return 0;
}
