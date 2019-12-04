#include <iostream>
#include <pcl/common/common.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/point_types.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/poisson.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <sstream>

void downsampling(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr &out_cloud, double resolution)
{
	pcl::VoxelGrid<pcl::PointXYZ> ds_filter;

	ds_filter.setInputCloud(in_cloud);
	ds_filter.setLeafSize(resolution, resolution, resolution);
	ds_filter.filter(*out_cloud);
}

void estimateReversedNormals(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in_cloud, pcl::PointCloud<pcl::PointNormal>::Ptr &out_cloud, double ne_radius)
{
	pcl::search::Search<pcl::PointXYZ>::Ptr tree;

	if (in_cloud->isOrganized()) {
		tree.reset(new pcl::search::OrganizedNeighbor<pcl::PointXYZ>());
	} else {
		tree.reset(new pcl::search::KdTree<pcl::PointXYZ>(false));
	}

	tree->setInputCloud(in_cloud);

	std::cout << "begin normal estimation" << std::endl;
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;

//	ne.setNumberOfThreads(8);
	ne.setInputCloud(in_cloud);

	ne.setSearchMethod(tree);
	ne.setRadiusSearch(ne_radius);	// default 0.01
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
	pcl::concatenateFields(*in_cloud, *cloud_normals, *out_cloud);

}

int main (int argc, char* argv[])
{
	if (argc != 6) {
		std::cout << "Usage: Command <input_file> <output_dir> <resolution(m)> <ne_radius(m)> <depth>" << std::endl;

		return -1;
	}

	std::string input_name(argv[1]);

	/* Size of each voxel for downsampling using voxel grid, smaller size -> more points remains. Default 0.02 */
	float resolution = std::atof(argv[3]);

	/* Radius for normal estimation. Default 0.06 */
	float ne_radius = std::atof(argv[4]);

	/* The deeper the depth is, more polygons remains. Default 6 */
	int depth = std::atoi(argv[5]);

	pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>);

	pcl::io::loadPLYFile<pcl::PointXYZ>(input_name, *in_cloud);

	// Step 1.5: Downsampling in_cloud
	pcl::PointCloud<pcl::PointXYZ>::Ptr down_sampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);

	downsampling(in_cloud, down_sampled_cloud, resolution);

	std::cout << "Number of downsampled points = " << down_sampled_cloud->size() << std::endl;

	// Step 2: Estimate normals
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_smoothed_normals(new pcl::PointCloud<pcl::PointNormal>);

	estimateReversedNormals(down_sampled_cloud, cloud_smoothed_normals, ne_radius);

	std::cout << "begin poisson reconstruction" << std::endl;
	pcl::Poisson<pcl::PointNormal> poisson;

	poisson.setDepth(depth);	// default 9
	poisson.setInputCloud(cloud_smoothed_normals);

	pcl::PolygonMesh mesh;

	poisson.reconstruct(mesh);

	std::cout << "Number of polygons = " << mesh.polygons.size() << std::endl;

	std::stringstream ss;

	ss << argv[2] << "possion.vtk";

	pcl::io::saveVTKFile(ss.str(), mesh);

	// Save to OBJ File

	ss.str("");

	ss << argv[2] << "poisson.obj";

	pcl::io::saveOBJFile(ss.str(), mesh);

   return 0;
}
