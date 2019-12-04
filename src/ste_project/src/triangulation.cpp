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

// Fill holes
#include <pcl/surface/vtk_smoothing/vtk_utils.h>
#include <vtkSmartPointer.h>
#include <vtkFillHolesFilter.h>
#include <vtkPolyData.h>

void downsampling(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr &out_cloud, double resolution)
{
	pcl::VoxelGrid<pcl::PointXYZ> ds_filter;

	ds_filter.setInputCloud(in_cloud);
	ds_filter.setLeafSize(resolution, resolution, resolution);
	ds_filter.filter(*out_cloud);
}

void segmentation(const pcl::PointCloud<pcl::PointXYZ>::Ptr &in_cloud, std::vector<pcl::PointIndices> &cluster_indices,
					pcl::PointCloud<pcl::Normal>::Ptr &normal_small_scale, pcl::PointCloud<pcl::Normal>::Ptr &normal_large_scale,
					double small_scale, double large_scale, double segradius)
{
	pcl::search::Search<pcl::PointXYZ>::Ptr tree;

	if (in_cloud->isOrganized()) {
		tree.reset(new pcl::search::OrganizedNeighbor<pcl::PointXYZ>());
	} else {
		tree.reset(new pcl::search::KdTree<pcl::PointXYZ>(false));
	}

	tree->setInputCloud(in_cloud);

	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;

	ne.setInputCloud(in_cloud);
	ne.setSearchMethod(tree);
	ne.setViewPoint(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());

	// Compute small scale normals
	ne.setRadiusSearch(small_scale);
	ne.compute(*normal_small_scale);

	// Compute large scale normals
	ne.setRadiusSearch(large_scale);
	ne.compute(*normal_large_scale);

	// Compute DoN
	pcl::PointCloud<pcl::PointNormal>::Ptr don_cloud(new pcl::PointCloud<pcl::PointNormal>);
	pcl::copyPointCloud(*in_cloud, *don_cloud);

	pcl::DifferenceOfNormalsEstimation<pcl::PointXYZ, pcl::Normal, pcl::PointNormal> don;

	don.setInputCloud(in_cloud);
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

	pcl::EuclideanClusterExtraction<pcl::PointNormal> ec;

	ec.setClusterTolerance(segradius);
	ec.setMinClusterSize(50);
	ec.setMaxClusterSize(1000000);
	ec.setSearchMethod(segtree);
	ec.setInputCloud(don_cloud);
	ec.extract(cluster_indices);
}

void createTriangulationMesh(const pcl::PointCloud<pcl::PointNormal>::Ptr &in_cloud, pcl::PolygonMesh &mesh,
								double radius)
{
	// Create search tree*
	pcl::search::KdTree<pcl::PointNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointNormal>);
	tree->setInputCloud(in_cloud);

	// Initialize objects
	pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;

	// Set the maximum distance between connected points (maximum edge length of a triangle)
	gp3.setSearchRadius(radius);			// Default 0.1

	// Set typical values for the parameters
	gp3.setMu(2.5);	// Default 2.5
	gp3.setMaximumNearestNeighbors(100);	// Default 100
	gp3.setMaximumSurfaceAngle(M_PI / 4); 	// Default M_PI/4 = 45 degrees
	gp3.setMinimumAngle(M_PI / 18); 		// Default M_PI/18 = 10 degrees
	gp3.setMaximumAngle(2 * M_PI / 3); 		// Default 2 * M_PI / 3 =  120 degrees
	gp3.setNormalConsistency(false);		// Default false

	// Get result
	gp3.setInputCloud(in_cloud);
	gp3.setSearchMethod(tree);
	gp3.reconstruct(mesh);
}


void fillHoles(const pcl::PolygonMesh &in_mesh, pcl::PolygonMesh &out_mesh, double hole_size)
{
	vtkSmartPointer<vtkPolyData> input;
	pcl::VTKUtils::mesh2vtk(in_mesh, input);

	vtkSmartPointer<vtkFillHolesFilter> fillHolesFilter = vtkSmartPointer<vtkFillHolesFilter>::New();

	fillHolesFilter->SetInputData(input);
	fillHolesFilter->SetHoleSize(hole_size);
	fillHolesFilter->Update();

	vtkSmartPointer<vtkPolyData> polyData = fillHolesFilter->GetOutput();

	pcl::VTKUtils::vtk2mesh(polyData, out_mesh);
}

int main(int argc, char **argv)
{
	// All distance are in meters
	if (argc != 8) {
		std::cout << "Usage: rosrun ste_project triangulation <in_ply_file> <out_folder> <resolution(m)> <small_scale(m)> <large_scale(m)> <segradius(m)> <triangulation_radius(m)>" << std::endl;
		exit(EXIT_FAILURE);
	}

	std::string in_ply_name(argv[1]);
	std::string out_dir_name(argv[2]);

	/* Size of a voxel in downsampling, larger means less points remain, default 0.02 */
	double resolution = std::atof(argv[3]);

	/* Small radius of the normal estimation, default 0.03 */
	double small_scale = std::atof(argv[4]);

	/* Large radius of the normal estimation, default 0.04 */
	double large_scale = std::atof(argv[5]);

	/* Threshold of the different of normals segmentation, default 0.05 */
	double segradius = std::atof(argv[6]);

	/* Maximum length of a triangle in triangulation, default 0.1 */
	double triangulation_rad = std::atof(argv[7]);

	pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>);

	// Step 1: Load from ply file to in_cloud
	pcl::io::loadPLYFile(in_ply_name, *in_cloud);
	std::cout << "Number of points = " << in_cloud->size() << std::endl;

	// Step 1.5: Downsampling in_cloud
	pcl::PointCloud<pcl::PointXYZ>::Ptr down_sampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);

	downsampling(in_cloud, down_sampled_cloud, resolution);

	// Step 2: Segmentation
	std::vector<pcl::PointIndices> cluster_indices;
	pcl::PointCloud<pcl::Normal>::Ptr normal_small_scale(new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<pcl::Normal>::Ptr normal_large_scale(new pcl::PointCloud<pcl::Normal>);

	segmentation(down_sampled_cloud, cluster_indices, normal_small_scale, normal_large_scale,
					small_scale, large_scale, segradius);

	int j = 0;

	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it, ++j) {
		pcl::PointCloud<pcl::Normal>::Ptr normal_cluster(new pcl::PointCloud<pcl::Normal>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normal(new pcl::PointCloud<pcl::PointNormal>);

		for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) {
			normal_cluster->push_back(normal_small_scale->points[*pit]);
			cloud_cluster->push_back(down_sampled_cloud->points[*pit]);
		}

		pcl::concatenateFields(*cloud_cluster, *normal_cluster, *cloud_with_normal);

		std::cout << "Number of points in cloud " << j << " is " << cloud_with_normal->size() << std::endl;
		// Triangulation
		pcl::PolygonMesh triangles;

		createTriangulationMesh(cloud_with_normal, triangles, triangulation_rad);

		std::stringstream ss;

		std::cout << "Triangle num = " << triangles.polygons.size() << std::endl;

		ss << out_dir_name << "mesh_" << j << ".vtk";

		pcl::io::saveVTKFile(ss.str(), triangles);

		// Fill hole
		pcl::PolygonMesh less_hole_triangles;

		fillHoles(triangles, less_hole_triangles, triangulation_rad * 10);

		ss.str("");

		ss << out_dir_name << "less_hole_mesh_" << j << ".vtk";

		pcl::io::saveVTKFile(ss.str(), less_hole_triangles);

		// Save to OBJ File

		ss.str("");

		ss << out_dir_name << "mesh_" << j << ".obj";

		pcl::io::saveOBJFile(ss.str(), triangles);

		ss.str("");

		ss << out_dir_name << "less_hole_mesh_" << j << ".obj";

		pcl::io::saveOBJFile(ss.str(), less_hole_triangles);
	}



	return 0;
}
