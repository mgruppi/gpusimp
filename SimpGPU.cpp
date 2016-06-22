#include "SimpGPU.h"
#include <algorithm>
#include <cmath>



//get position from header array
//Size of each structure on arrays
const int HEADER_SIZE = 3;
const int FACE_SIZE = 3;
const int VERTEX_SIZE = 3;
const int FACE_DATA_BATCH_SIZE = 50;
const int EDGE_DATA_BATCH_SIZE = 50;
const int QUADRIC_SIZE = 16; //Quadric for a vertex is a 4x4 matrix
const int EDGE_SIZE = 2;


//ACCESS DEFINES
#define getFaceVertexId(face,vertex) h_faces[FACE_SIZE*face+vertex]
#define getFaceHeaderPos(vertex) h_vert_face_header[HEADER_SIZE*vertex]
#define getFaceCurrSize(vertex) h_vert_face_header[HEADER_SIZE*vertex+1]
#define faceIncreaseSize(vertex) h_vert_face_header[HEADER_SIZE*vertex+1]++
#define getFaceId(vid,p) h_vert_face_data[getFaceHeaderPos(vid)+p]
#define getX(vid) h_vertices[VERTEX_SIZE*vid]
#define getY(vid) h_vertices[VERTEX_SIZE*vid+1]
#define getZ(vid) h_vertices[VERTEX_SIZE*vid+2]

//ACCESS TO EDGES
#define getEdgeVertexId(edge,vid) h_edges[EDGE_SIZE*edge+vid] //Get vertex id (0 or 1) of edge
#define getEdgeHeaderPos(vid) h_vert_edge_header[HEADER_SIZE*vid]
#define getEdgeCurrSize(vid) h_vert_edge_header[HEADER_SIZE*vid+1]
#define edgeIncreaseSize(vid) h_vert_edge_header[HEADER_SIZE*vid+1]++
#define getEdgeDataId(vid,p) h_vert_edge_data[getEdgeHeaderPos(vid)+p]

//Operations
#define getPlacementX(vid1,vid2) ((getX(vid1)+getX(vid2))/2
#define getPlacementY(vid1,vid2) ((getY(vid1)+getY(vid2))/2
#define getPlacementZ(vid1,vid2) ((getZ(vid1)+getZ(vid2))/2

SimpGPU::SimpGPU(Surface* so)
{
  s = so;

}

void SimpGPU::simplify(int goal, int gridres=1)
{
  cerr << "Initializing Data Structures...\n";
  initDataStructures();
  cerr << "Computing initial quadrics...\n";
  initQuadrics();
  cerr << "Computing edges...\n";
  initEdges();

  updateSurface();

}

void SimpGPU::initDataStructures()
{
  n_faces = s->m_faces.size();
  n_vertices = s->m_points.size();
  h_faces.resize(FACE_SIZE*n_faces);
  h_face_removed.resize(n_faces);
  h_vertices.resize(VERTEX_SIZE*n_vertices);
  h_vertex_removed.resize(n_vertices);
  h_quadrics.resize(16*n_vertices);


  h_vert_face_header.resize(HEADER_SIZE*n_vertices);
  h_vert_face_data.resize(FACE_DATA_BATCH_SIZE*n_vertices);

  //TODO:Read surface in this format.
  for(int i = 0; i < s->m_points.size(); ++i)
  {
    h_vertices[3*i] = s->m_points[i]->x;
    h_vertices[3*i+1] = s->m_points[i]->y;
    h_vertices[3*i+2] = s->m_points[i]->z;
    h_vertex_removed[i] = false;
  }
  for(int i = 0 ; i < s->m_faces.size(); ++i)
  {
    h_faces[3*i] = s->m_faces[i]->points[0]->id;
    h_faces[3*i+1] = s->m_faces[i]->points[1]->id;
    h_faces[3*i+2] = s->m_faces[i]->points[2]->id;
    h_face_removed[i] = false;
  }


  timespec tinit, tinit0, tinit1;
  gettime(tinit0);
  //TODO: Create vert_face on device.
  int max_size = 0;
  for(int i = 0; i < n_vertices; ++i)
  {
    int vfaces=0;
    h_vert_face_header[HEADER_SIZE*i] = i*FACE_DATA_BATCH_SIZE;
    h_vert_face_header[HEADER_SIZE*i+1]= 0; //size initially 0
    h_vert_face_header[HEADER_SIZE*i+2]=-1; //initially null (continuesTo)
  }
  //For the three vertices of a face, add face to their data array.
  for(int j = 0; j < n_faces; ++j)
  {
      //Add face to data array and increase size of vid in header
      //v0
      {
      int vid = getFaceVertexId(j,0);
      int pos = getFaceHeaderPos(vid);
      int curr_size = getFaceCurrSize(vid);
      h_vert_face_data[pos+curr_size] = j;
      faceIncreaseSize(vid);
      }

      //v1
      {
      int vid = getFaceVertexId(j,1);
      int pos = getFaceHeaderPos(vid);
      int curr_size = getFaceCurrSize(vid);
      h_vert_face_data[pos+curr_size] = j;
      faceIncreaseSize(vid);
      }

      //v2
      {
      int vid = getFaceVertexId(j,2);
      int pos = getFaceHeaderPos(vid);
      int curr_size = getFaceCurrSize(vid);
      h_vert_face_data[pos+curr_size] = j;
      faceIncreaseSize(vid);
      }
  }

  gettime(tinit1);
  tinit = diff(tinit0,tinit1);

  cerr << "Time to init data structures: " << getMilliseconds(tinit) << endl;
}

void SimpGPU::initEdges()
{

  cerr << "Init edges.\n";
  timespec te, te0, te1;

  gettime(te0);
  h_edges.resize(n_faces*6);//Estimate an initial size for edge vector
  h_vert_edge_header.resize(HEADER_SIZE*n_vertices);
  h_vert_edge_data.resize(EDGE_DATA_BATCH_SIZE*n_vertices);

  //init HEADER array
  for(int i = 0; i < n_vertices; ++i)
  {
    h_vert_edge_header[HEADER_SIZE*i] = EDGE_DATA_BATCH_SIZE*i;
    h_vert_edge_header[HEADER_SIZE*i+1] = 0;
    h_vert_edge_header[HEADER_SIZE*i+2] = -1;
  }

  int eid = 0;
  for(int i = 0; i < n_faces; ++i)
  {
    std::vector<int> vid;
    vid.push_back(getFaceVertexId(i,0));
    vid.push_back(getFaceVertexId(i,1));
    vid.push_back(getFaceVertexId(i,2));
    //We'll create only half-edges such that id(v0) < id(v1)
    //This is reasonable since the placement vertex is always at the edges's midpoint
    //Sort vector vid so ids are in ascending order

    //CHANGED FROM CPU
    std::sort(vid.begin(),vid.end());

    int w[3][2] = {{0,1},{0,2},{1,2}};

    for(int i = 0; i < 3; ++i)
    {
      bool found_edge = false;
      cerr << "eid: " << eid << " | " << vid[w[i][0]] << "," << vid[w[i][1]] << endl;

      for(int j = 0; j < getEdgeCurrSize(vid[w[i][0]]); ++j)
      {
        //Check wheter edge already exists
        if(getEdgeVertexId(getEdgeDataId(vid[w[i][0]],j),1) == vid[w[i][1]])
          found_edge = true;
      }
      if(!found_edge)
      {
        h_edges[EDGE_SIZE*eid] = vid[w[i][0]];
        h_edges[EDGE_SIZE*eid+1] = vid[w[i][1]];
        cerr << "adding " << eid << " to " << getEdgeHeaderPos(vid[w[i][0]])+getEdgeCurrSize(vid[w[i][0]]) << endl;
        h_vert_edge_data[getEdgeHeaderPos(vid[w[i][0]])+getEdgeCurrSize(vid[w[i][0]])] = eid;
        cerr << "added: " << h_vert_edge_data[getEdgeHeaderPos(vid[w[i][0]])+getEdgeCurrSize(vid[w[i][0]])] << endl;
        //h_vert_edge_data[getEdgeHeaderPos(vid[w[i][1])+getEdgeCurrSize(vid[w[i][1]])]]=eid;
        edgeIncreaseSize(vid[w[i][0]]);
        //edgeIncreaseSize(vid[w[i][1]]);
        eid++;
      }
    }


    //TODO: check if edge exists and do not add it
  }
  gettime(te1);
  te = diff(te0,te1);
  cerr << "Time to init edges: " << getMilliseconds(te) << endl;
  cerr << "No. of edges: " << eid <<endl;
}

void SimpGPU::initQuadrics()
{
  //Quadric Q (4x4 matrix) is the sum of all planes tangent to a vertex v (Garland, 97).
//Get planes of every vertex faces
  timespec tq, tq0, tq1;
  gettime(tq0);
  for(int i = 0 ; i < n_vertices; ++i)
  {
      double Kp[16];
      for(int j = 0; j < getFaceCurrSize(i); ++j)
      {
        int fid = getFaceId(i,j);
        int v0 = getFaceVertexId(fid,0);
        int v1 = getFaceVertexId(fid,1);
        int v2 = getFaceVertexId(fid,2);
        //Calculate vectors v0v1 and v0v2 for face fid
        double v0v1x = getX(v1) - getX(v0);
        double v0v1y = getY(v1) - getY(v0);
        double v0v1z = getZ(v1) - getZ(v0);

        double v0v2x = getX(v2) - getX(v0);
        double v0v2y = getY(v2) - getY(v0);
        double v0v2z = getZ(v2) - getZ(v0);

        //cross product v0v1 v0v2
        double vvx = v0v1y*v0v2z - v0v1z*v0v2y;
        double vvy = v0v1z*v0v2x - v0v1x*v0v2z;
        double vvz = v0v1x*v0v2y - v0v1y*v0v2x;

        //normalize vv
        double mag = sqrt(vvx*vvx + vvy*vvy + vvz*vvz);
        vvx = vvx/mag;
        vvy = vvy/mag;
        vvz = vvz/mag;
        //Apply v0 to find parameter d of plane equation
        double d = vvx*getX(v0) + vvy*getY(v0);
        d += vvz*getZ(v0);
        d*=-1;
        double plane_eq[4]={vvx,vvy,vvz,d};
        //For this plane, the fundamental quadric Kp is the product of vectors plane_eq and plane_eq(transposed) (garland97)
        for(int k = 0; k < 4; ++k)
        {
          for(int l = 0; l < 4; ++l)
          {
            Kp[4*k+l] = plane_eq[k]*plane_eq[l];
          }
        }
        //Add Kp to i's quadric here.
        quadricAdd(h_quadrics.data()+(QUADRIC_SIZE*i),Kp);

      }
  }

  gettime(tq1);
  tq = diff(tq0,tq1);
  cerr << "Time to initialize quadrics: " << getMilliseconds(tq) << endl;
}

double SimpGPU::getCost(int eid)
{
  double tempQ[16] = {0}; //Temporary placement quadric
  quadricCopy(tempQ,h_quadrics.data()+(QUADRIC_SIZE*getEdgeVertexId(eid,0)));
  quadricAdd(tempQ,h_quadrics.data()+(QUADRIC_SIZE*getEdgeVertexId(eid,1)));
  double tempV[4];//TEmporary vertex placement
  tempV[0] = getPlacementX(getEdgeVertexId(eid,0), getEdgeVertexId(eid,1)));
  tempV[1] = getPlacementY(getEdgeVertexId(eid,0), getEdgeVertexId(eid,1)));
  tempV[2] = getPlacementZ(getEdgeVertexId(eid,0), getEdgeVertexId(eid,1)));
  tempV[3] = 1;

  //calculate error (vQvT)
  double vQ[4] = {0};
  double cost = 0;
  for(int k = 0; k < 4; ++k)
  {
    for( int l = 0; l < 4; l++)
    {
      vQ[k] += tempV[l]*tempQ[4*k+l];
    }
  }

  for(int i = 0; i < 4; ++i)
  {
    cost += vQ[i]*tempV[i];
  }

  return cost;

}

//Update surface for output purposes ONLY
void SimpGPU::updateSurface()
{
  //Converting points back
  for(int i = 0 ; i < s->m_points.size(); ++i)
  {
    s->m_points[i]->x = getX(i);
    s->m_points[i]->y = getY(i);
    s->m_points[i]->z = getZ(i);
    s->m_points[i]->removed = (h_vertex_removed[i] || getFaceCurrSize(i)==0); //If a vertex has no face, remove it too
  }

  //Converting faces back
  for(int j = 0; j < s->m_faces.size(); ++j)
  {
    s->m_faces[j]->points[0] = s->m_points[getFaceVertexId(j,0)];
    s->m_faces[j]->points[1] = s->m_points[getFaceVertexId(j,1)];
    s->m_faces[j]->points[2] = s->m_points[getFaceVertexId(j,2)];
    s->m_faces[j]->removed = h_face_removed[j];
  }
}