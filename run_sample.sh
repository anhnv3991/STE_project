rosrun ste_project triangulation data/point_cloud/book_and_fruit/reconstruction_sequential/scene_dense.ply result/books_and_fruit/greedy_triangulation/ 0.02 0.03 0.04 0.05 0.1

rosrun ste_project triangulation data/point_cloud/car_toy/reconstruction_sequential/scene_dense.ply result/car_toy/greedy_triangulation/ 0.04 0.05 0.06 0.08 0.1

rosrun ste_project triangulation data/point_cloud/robot/reconstruction_sequential/scene_dense.ply result/robot/greedy_triangulation/ 0.04 0.05 0.07 0.09 0.1

rosrun ste_project poisson data/point_cloud/book_and_fruit/reconstruction_sequential/scene_dense.ply result/books_and_fruit/poisson/ 0.02 0.06 6

rosrun ste_project poisson data/point_cloud/car_toy/reconstruction_sequential/scene_dense.ply result/car_toy/poisson/ 0.02 0.03 6

rosrun ste_project poisson data/point_cloud/robot/reconstruction_sequential/scene_dense.ply result/robot/poisson/ 0.02 0.04 6

