#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

// For creating convex hulls
#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/surface/convex_hull.h>

// For extract subset of a point cloud
#include <pcl/filters/extract_indices.h>

#include <string>

#define MIN_PERCENTAGE_ (0.1)


int main(int argc, char *argv[])
{
	if (argc != 2) {
		std::cout << "Usage: command <input_ply_file>" << std::endl;
		return -1;
	}

	std::string input_name(argv[1]);
	pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>);

	// Load the input cloud
	pcl::io::loadPLYFile(input_name, *in_cloud);

	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

	pcl::SACSegmentation<pcl::PointXYZ> seg;

	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setDistanceThreshold(0.01);

	int input_size = in_cloud->size();

	int seg_num = 0;

	while (in_cloud->size() > input_size * MIN_PERCENTAGE_ / 100.0) {
		seg.setInputCloud(in_cloud);
		seg.segment(*inliers, *coefficients);

		// Extract the current plane
		pcl::PointCloud<pcl::PointXYZ>::Ptr remain_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::ExtractIndices<pcl::PointXYZ> extract;

		extract.setInputCloud(in_cloud);
		extract.setIndices(inliers);
		extract.setNegative(false);

		pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZ>);

		extract.filter(*out_cloud);

		// Save to a pcd file
		std::stringstream out_pcd_name;

		out_pcd_name << "cloud_" << seg_num << ".pcd";

		pcl::io::savePCDFile(out_pcd_name.str(), *out_cloud);

		// Convert to a convex hull
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::ConvexHull<pcl::PointXYZ> chull;
		pcl::PolygonMesh mesh;

		chull.setInputCloud(out_cloud);
		chull.reconstruct(*cloud_hull, mesh.polygons);

		pcl::toPCLPointCloud2(*cloud_hull, mesh.cloud);

		std::stringstream out_fname;

		out_fname << "seg_" << seg_num << ".vtk";

		pcl::io::saveVTKFile(out_fname.str(), mesh);

		++seg_num;

		// Get the remaining cloud
		extract.setNegative(true);	// Output is the old cloud excluding inliers
		extract.filter(*remain_cloud);
		in_cloud.swap(remain_cloud);
	}


	return 0;
}
