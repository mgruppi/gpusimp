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


class SimpGPU
{

public:
  SimpGPU(Surface* so);

  //init
  void initDataStructures();
  void initEdges();
  void initQuadrics();
  void initUniformGrid();

  //Simplification
  //double getCost(int eid); //Get cost form edge eid
  double updateCosts(int vid);//Update costs for vid's edges
  void simplify(int, int);

  //Etc
  void updateSurface();

  //Host
  Surface* s;
  int n_faces, n_vertices;





};

#endif // _SIMPGPU_H__
