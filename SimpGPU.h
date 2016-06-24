#ifndef _SIMPGPU_H__
#define _SIMPGPU_H__
#include "Surface.h"
#include <iostream>
#include "Classes.h"
#include "kernel.h"
using namespace std;

//Thrust includes
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


//HOST MEMBERS
//VERTICES
thrust::host_vector<double> h_vertices; //Every 3 positions of this vector is a Vertex (x,y,z)
thrust::host_vector<bool> h_vertex_removed;
thrust::host_vector<double> h_quadrics; //Quadrics for vertex i
//FACES
thrust::host_vector<int> h_faces; // Every 3 positions of this vector is a face (p0,p1,p2)
thrust::host_vector<int> h_vert_face_header; //[DATA_POSITION,DATA_SIZE,CONTINUES_TO]
thrust::host_vector<int> h_vert_face_data; //[FACE_ID,...,FACE_ID]
thrust::host_vector<bool> h_face_removed;
//EDGES
thrust::host_vector<int> h_edges; //Every 2 positions is a half-edge [vfrom,vto]
thrust::host_vector<double> h_edge_cost; //Cost of edge i
thrust::host_vector<int> h_edge_queue; //Queue of edges sorted by cost
thrust::host_vector<int> h_vert_edge_header; //[EDGE_DATA_POSITION,EDGE_DATA_SIZE,CONTINUES_TO]
thrust::host_vector<int> h_vert_edge_data; //[EDGE_ID, ..., EDGE_ID]
thrust::host_vector<bool> h_edge_removed; //bool


class SimpGPU
{

public:
  SimpGPU(Surface* so);

  //init
  void initDataStructures();
  void initEdges();
  void initQuadrics();

  //Simplification
  double getCost(int eid); //Get cost form edge eid
  double updateCosts(int vid);//Update costs for vid's edges
  void simplify(int, int);

  //Etc
  void updateSurface();

  //Host
  Surface* s;
  int n_faces, n_vertices, n_edges;




};

#endif // _SIMPGPU_H__
