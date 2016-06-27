#include "SimpGPU.h"
#include <algorithm>
#include <cmath>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>



//TIME
long time_collapse = 0;

//get position from header array
//Size of each structure on arrays
const int HEADER_SIZE = 3;
const int FACE_SIZE = 3;
const int VERTEX_SIZE = 3;
const int FACE_DATA_BATCH_SIZE = 50;
const int EDGE_DATA_BATCH_SIZE = 50;
const int QUADRIC_SIZE = 16; //Quadric for a vertex is a 4x4 matrix
const int EDGE_SIZE = 2;

//UGRID
const int CELL_HEADER_SIZE = 2;
const int EDGE_HEADER_SIZE = 2;


//ACCESS Vertex
#define getFaceVertexId(face,vertex) h_faces[FACE_SIZE*face+vertex]
#define getFaceHeaderPos(vertex) h_vert_face_header[HEADER_SIZE*vertex]
#define getFaceCurrSize(vertex) h_vert_face_header[HEADER_SIZE*vertex+1]
#define faceIncreaseSize(vertex) h_vert_face_header[HEADER_SIZE*vertex+1]++
#define faceDecreaseSize(vertex) h_vert_face_header[HEADER_SIZE*vertex+1]--
#define getFaceId(vid,p) h_vert_face_data[getFaceHeaderPos(vid)+p]
#define getX(vid) h_vertices[VERTEX_SIZE*vid]
#define getY(vid) h_vertices[VERTEX_SIZE*vid+1]
#define getZ(vid) h_vertices[VERTEX_SIZE*vid+2]

//ACCESS TO EDGES
#define getEdgeVertexId(edge,vid) h_edges[EDGE_SIZE*edge+vid] //Get vertex id (0 or 1) of edge
#define getEdgeFromHeaderPos(vid) h_vert_edge_from_header[HEADER_SIZE*vid]
#define getEdgeFromCurrSize(vid) h_vert_edge_from_header[HEADER_SIZE*vid+1]
#define edgeFromIncreaseSize(vid) h_vert_edge_from_header[HEADER_SIZE*vid+1]++
#define edgeFromDecreaseSize(vid) h_vert_edge_from_header[HEADER_SIZE*vid+1]--
#define getEdgeFromDataId(vid,p) h_vert_edge_from_data[getEdgeFromHeaderPos(vid)+p]

//edge_to
#define edgeToIncreaseSize(vid) h_vert_edge_to_header[HEADER_SIZE*vid+1]++
#define edgeToDecreaseSize(vid) h_vert_edge_to_header[HEADER_SIZE*vid+1]--
#define getEdgeToHeaderPos(vid) h_vert_edge_to_header[HEADER_SIZE*vid]
#define getEdgeToCurrSize(vid) h_vert_edge_to_header[HEADER_SIZE*vid+1]
#define getEdgeToDataId(vid,p) h_vert_edge_to_data[getEdgeToHeaderPos(vid)+p]

//Access
#define getPlacementX(vid1,vid2) (getX(vid1)+getX(vid2))/2
#define getPlacementY(vid1,vid2) (getY(vid1)+getY(vid2))/2
#define getPlacementZ(vid1,vid2) (getZ(vid1)+getZ(vid2))/2

//UNIFORM GRID
#define getCellHeaderPos(cell) h_cell_header[CELL_HEADER_SIZE*cell]
#define getCellHeaderSize(cell) h_cell_header[CELL_HEADER_SIZE*cell+1]
#define increaseCellSize(cell) h_cell_header[CELL_HEADER_SIZE*cell+1]++
#define getCellVertexId(cell,p) h_cell_data[getCellHeaderPos(cell)+p]


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
thrust::host_vector<int> h_vert_edge_from_header; //[EDGE_DATA_POSITION,EDGE_DATA_SIZE,CONTINUES_TO]
thrust::host_vector<int> h_vert_edge_from_data; //[EDGE_ID, ..., EDGE_ID]
thrust::host_vector<int> h_vert_edge_to_header;
thrust::host_vector<int> h_vert_edge_to_data;
thrust::host_vector<bool> h_edge_removed; //bool

//UNIFORM GRID
int n_cells;

//VERTEX
thrust::host_vector<int> h_cell_data; //contains id of vertices inside it
thrust::host_vector<int> h_cell_header; //Cell header containing initial position and size in h_cell vector
thrust::host_vector<int> h_vertex_in_cell;
thrust::host_vector<int> h_initial_vertices; //Number of initial vertices in each cell


//EDGE
thrust::host_vector<int> h_cell_queue; //Queue of edges
thrust::host_vector<int> h_cell_queue_size; //Size of queues
thrust::host_vector<int> h_prev_queue_size;




//UNIFORM GRID DIM and OFFSET
double dim[3];
double offset[3];
int grid_res;

SimpGPU::SimpGPU(Surface* so)
{
  s = so;

}


//Edge comparison function
//Edge queue is sorted so that the least cost is in the last position
bool compareEdges(int left, int right)
{
  return h_edge_cost[left] > h_edge_cost[right];
}

bool isEntirelyInCell(int eid)
{
  //Return if both endpoints are withing the same cell
  return h_vertex_in_cell[getEdgeVertexId(eid,0)] == h_vertex_in_cell[getEdgeVertexId(eid,1)];
}

bool isCrownInCell(int vid)
{
  //Test edges leaving (from) vid to see if their endpoints also lie within vid's cell
  for(int i = 0 ; i < getEdgeFromCurrSize(vid); ++i)
  {
    //Check edge i from vid
    int eid = getEdgeFromDataId(vid,i);
    int endp = getEdgeVertexId(eid,1);
    if(h_vertex_in_cell[endp] != h_vertex_in_cell[vid])
      return false;
  }

  //Test edges arriving (to) vid
  for(int i = 0 ; i < getEdgeFromCurrSize(vid); ++i)
  {
    int eid = getEdgeToDataId(vid,i);
    int endp = getEdgeVertexId(eid,0);
    if(h_vertex_in_cell[endp] != h_vertex_in_cell[vid])
      return false;
  }

  return true;

  //TODO:add edges_to

}


double getCost(int eid)
{
  double tempQ[16] = {0}; //Temporary placement quadric
  quadricCopy(tempQ,h_quadrics.data()+(QUADRIC_SIZE*getEdgeVertexId(eid,0)));
  quadricAdd(tempQ,h_quadrics.data()+(QUADRIC_SIZE*getEdgeVertexId(eid,1)));

  //cerr <<"\ntempQ ";
  //for(int i = 0; i < 16; ++i) cerr << tempQ[i] << " ";
  //cerr << endl;

  double tempV[4];//TEmporary vertex placement
  tempV[0] = getPlacementX(getEdgeVertexId(eid,0), getEdgeVertexId(eid,1));
  tempV[1] = getPlacementY(getEdgeVertexId(eid,0), getEdgeVertexId(eid,1));
  tempV[2] = getPlacementZ(getEdgeVertexId(eid,0), getEdgeVertexId(eid,1));
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

//Initialize uniform grid
void SimpGPU::initUniformGrid()
{
  timespec tu, tu0, tu1;
  gettime(tu0);

  n_cells = grid_res*grid_res*grid_res;
  cerr << "No. of cells: " << n_cells << endl;

  //Update offset and dim
  offset[0] = s->bbox.minx;
  offset[1] = s->bbox.miny;
  offset[2] = s->bbox.minz;

  dim[0] = s->bbox.getXLen()/grid_res;
  dim[1] = s->bbox.getYLen()/grid_res;
  dim[2] = s->bbox.getZLen()/grid_res;

  //Deprecated
  h_vertex_in_cell.resize(n_vertices);
  h_cell_data.resize(n_vertices);
  h_cell_header.resize(n_cells*CELL_HEADER_SIZE);

  h_initial_vertices.resize(n_cells);
  h_cell_queue.resize(n_edges*n_cells);
  h_cell_queue_size.resize(n_cells);
  h_prev_queue_size.resize(n_cells);

  for(int i = 0; i < n_cells; ++i)
  {
    h_initial_vertices[i] = 0;
    h_cell_queue_size[i] = 0;
  }


  //Compute gridcell for each vertex
  //h_vertex_in_cell stores the cell in which vertex i is located
  for(int i = 0; i < n_vertices; ++i)
  {
    int cx = (getX(i) - offset[0])/dim[0];
    cx -= cx/grid_res;
    int cy = (getY(i) - offset[1])/dim[1];
    cy -= cy/grid_res;
    long long cz = (getZ(i) - offset[2])/dim[2];
    cz -= cz/grid_res;
    int cpos = cx + grid_res*cy + grid_res*grid_res*cz;
    //unsigned int cpos = getGridCell(i);
    //cerr << "vertex " << i << " cell " << cpos << endl;
    //Add vertex to cell
    h_vertex_in_cell[i] = cpos;
    h_initial_vertices[cpos]++;
    //cerr << "add to " << cpos*n_vertices+h_cell_header[cpos*CELL_HEADER_SIZE+1] << endl;
    //h_cell[cpos*n_vertices+h_cell_header[cpos*CELL_HEADER_SIZE+1]] = i;
    increaseCellSize(cpos);
  }


  //update headers with size of each cell
  //Every cell has a initial position and a size in h_edge_data array
  h_cell_header[0] = 0;
  h_cell_header[1] = 0;
  for(int i = 1; i < n_cells;++i)
  {
    //set (pos) pointer
    h_cell_header[i*CELL_HEADER_SIZE] = h_initial_vertices[i-1]+h_cell_header[(i-1)*CELL_HEADER_SIZE];
    h_cell_header[i*CELL_HEADER_SIZE+1] = 0;
    //cerr << "Cell " << i <<  " pos " << h_cell_header[i*CELL_HEADER_SIZE] << endl;
  }

  //TODO: Sort h_cell_data containing every vertex id according to vector h_vertex_in_cell
  //We finally add vertex ids to h_data_array
  for(int i = 0; i < n_vertices; ++i)
  {
    //cerr << "Adding " << i << " to " << h_vertex_in_cell[i] << " size " << getCellHeaderSize(h_vertex_in_cell[i]) << endl;
    //add vertex to cell data
    h_cell_data[getCellHeaderPos(h_vertex_in_cell[i]) + getCellHeaderSize(h_vertex_in_cell[i])] = i;
    increaseCellSize(h_vertex_in_cell[i]);
  }

  //Now we compute edge queues for each cell
  //An edge will be added to queue iff it is entirely in cell and both endpoints have crown entirely in the same cell
  for(int i = 0; i < n_cells;++i)
  {
    //For every vertex in the cell, we get its edge
    for(int j = 0; j < getCellHeaderSize(i); ++j)
    {
      //For every edge of vertex j, we check if its eligible to go into queue
      for(int k = 0; k < getEdgeFromCurrSize(getCellVertexId(i,j)); ++k)
      {
        int eid = getEdgeFromDataId(getCellVertexId(i,j),k);

        //cerr << "edge " << eid << " in " << i << " - edges: " << endl;
        if(isEntirelyInCell(eid) && isCrownInCell(getEdgeVertexId(eid,0)) && isCrownInCell(getEdgeVertexId(eid,1)))
        {
          //cerr << "YES\n";
          h_cell_queue[i*n_cells+h_cell_queue_size[i]] = eid;
          h_cell_queue_size[i]++;

        }
      }
    }
  }

  //sort queues
  for(int i = 0; i < n_cells; ++i)
  {
    std:sort(h_cell_queue.data()+(i*n_cells), h_cell_queue.data()+(i*n_cells+h_cell_queue_size[i]),compareEdges);
    //h_prev_queue_size[i] = h_cell_queue_size[i];
  }

  // cerr << "Cells: " << endl;
  // for (int i = 0 ; i < n_cells; ++i)
  // {
  //   cerr << i << "-->> ";
  //   for (int j = 0 ; j < h_cell_queue_size[i]; ++j)
  //   {
  //     cerr << h_cell_queue[i*n_cells+j] << " ";
  //   }
  //   cerr << endl;
  // }
  //


  gettime(tu1);
  tu = diff(tu0,tu1);
  cerr << "Time to initialize uniform grid: " << getMilliseconds(tu) << endl;

}


void removeEdge(int eid)
{
  int v1 = getEdgeVertexId(eid,0);
  int v2 = getEdgeVertexId(eid,1);

  //Remove from

  for(int i = 0 ; i < getEdgeFromCurrSize(v1); ++i)
  {
    if(h_vert_edge_from_data[getEdgeFromHeaderPos(v1)+i] == eid)
    {

      h_vert_edge_from_data[getEdgeFromHeaderPos(v1)+i] = h_vert_edge_from_data[getEdgeFromHeaderPos(v1)+getEdgeFromCurrSize(v1)-1];
      edgeFromDecreaseSize(v1);
      break;
    }
  }

  //Remove to
  for(int i = 0; i < getEdgeToCurrSize(v2);++i)
  {
    //cerr << "searching... ";
    //cerr << getEdgeToDataId(v2,i);
    if(h_vert_edge_to_data[getEdgeToHeaderPos(v2)+i] == eid)
    {
      //cerr <<"Found\n";
      h_vert_edge_to_data[getEdgeToHeaderPos(v2)+i] = h_vert_edge_to_data[getEdgeToHeaderPos(v2)+getEdgeToCurrSize(v2)-1];
      edgeToDecreaseSize(v2);
      break;
    }
  }

    h_edge_removed[eid] = true;

}

void collapse(int eid)
{
  int v1 = getEdgeVertexId(eid,0);
  int v2 = getEdgeVertexId(eid,1);

  //cin.get();

  cerr << "*********Collapsing " << eid << ": "<< v1 << " -> " << v2 << " = " << h_edge_cost[eid] <<  endl;

  //Iterate FACES
  //We are going to remove faces containing v1 and v2 simultaneously
  //The remaining faces from v1 are going to be moved to v2's list
  for(int i = 0; i < getFaceCurrSize(v1); ++i)
  {
    bool removeFace = false;
    int face_it = getFaceId(v1,i);
    //cerr << "Face " << face_it << endl;


    if(getFaceVertexId(face_it,0) == v2 || getFaceVertexId(face_it,1)==v2 || getFaceVertexId(face_it,2) == v2)
    {
      //cerr << "remove\n";
      //REMOVEFACE()
      //Remove this face since it shares v1 and v2
      //Warning: check if it is already removed?
      h_face_removed[face_it] = true;
      //Remove face from its vertices lists
      for(int j = 0 ; j < 3; ++j)
      {
        int vid = getFaceVertexId(face_it,j);
        for(int k = 0; k < getFaceCurrSize(vid);++k)
        {
          if(getFaceId(vid,k) == face_it){
            //cerr << getFaceId(getFaceVertexId(face_it,j),k) << " ";
            h_vert_face_data[vid*FACE_DATA_BATCH_SIZE+k] = h_vert_face_data[vid*FACE_DATA_BATCH_SIZE+getFaceCurrSize(vid)-1];
            faceDecreaseSize(vid);
          }
        }
      }

      //END OF REMOVEFACE()

    }
    else
    {
      //cerr <<"Update\n";
      //This face won't be removed, but modified
      //Find pointer to v1 and replace for v2
      if(h_faces[FACE_SIZE*face_it] == v1) h_faces[FACE_SIZE*face_it] = v2;
      else if (h_faces[FACE_SIZE*face_it+1] == v1) h_faces[FACE_SIZE*face_it+1] = v2;
      else if (h_faces[FACE_SIZE*face_it+2] == v1) h_faces[FACE_SIZE*face_it+2] = v2;
    }
  }
  //Copy list of faces from v1 to v2
  for(int i = 0; i < getFaceCurrSize(v1); ++i)
  {
    //Add face to v2's list
    h_vert_face_data[v2*FACE_DATA_BATCH_SIZE+getFaceCurrSize(v2)] = getFaceId(v1,i);
    faceIncreaseSize(v2);
  }

  h_vertices[VERTEX_SIZE*v2] = getPlacementX(v1,v2);
  h_vertices[VERTEX_SIZE*v2+1] = getPlacementY(v1,v2);
  h_vertices[VERTEX_SIZE*v2+2] = getPlacementZ(v1,v2);

  //REMOVEEDGE STARTS

  cerr << "Removing edges\n";

  //Update edges from v1
  removeEdge(eid);




  //Remove DOUBLE EDGES
  //Edges of which destiny already exists in v2
//  cerr << "v1 " << v1 << " - edges " << getEdgeFromCurrSize(v1) << endl;
  //cerr << "v2 " << v2 << " - edges " << getEdgeFromCurrSize(v2) << endl;
  for(int i = 0; i < getEdgeFromCurrSize(v1); ++i)
  {
    int eit = h_vert_edge_from_data[getEdgeFromHeaderPos(v1)+i];
    //Update <from> pointer to v2
    //h_edges[eit*EDGE_SIZE] = v2;
    //cerr << "eid " << eit << " is " << h_edges[eit*EDGE_SIZE] << " - " << h_edges[eit*EDGE_SIZE+1] << endl;

    for(int j = 0; j < getEdgeFromCurrSize(v2); ++j)
    {
      int eit2 = h_vert_edge_from_data[getEdgeFromHeaderPos(v2)+j];
      if(getEdgeVertexId(eit,1) == getEdgeVertexId(eit2,1) || getEdgeVertexId(eit,1) == v2)
      {
        cerr <<"Removing " << eit << endl;
        removeEdge(eit);
        j--;
        i--;
        break;
      }
    }
  }

  //IMPLEMENT TO_EDGE removal

  //Edges of which source already exists in v2
  //cerr << "v1 " << v1 << " - edges " << getEdgeToCurrSize(v1) << endl;
  //cerr << "v2 " << v2 << " - edges " << getEdgeToCurrSize(v2) << endl;
  for(int i = 0; i < getEdgeToCurrSize(v1); ++i)
  {
    //cin.get();
    int eit = h_vert_edge_to_data[getEdgeToHeaderPos(v1)+i];
    //Update <to> pointer to v2
    //h_edges[eit*EDGE_SIZE+1] = v2;
    //cerr << "eid--- " << eit << " is " << h_edges[eit*EDGE_SIZE] << " - " << h_edges[eit*EDGE_SIZE+1] << endl;

    for(int j = 0; j < getEdgeToCurrSize(v2); ++j)
    {
      int eit2 = h_vert_edge_to_data[getEdgeToHeaderPos(v2)+j];
      //cerr << "eid2- " << eit2 << " is " << h_edges[eit2*EDGE_SIZE] << " - " << h_edges[eit2*EDGE_SIZE+1] << endl;
      if(getEdgeVertexId(eit,0) == getEdgeVertexId(eit2,0) || getEdgeVertexId(eit,0) == v2)
      {
        cerr <<"Removing " << eit << endl;
        removeEdge(eit);
        j--;
        i--;
        break;
      }
    }
  }

  cerr << "Append edges\n";
  //Append edge lists
  for(int i = 0 ; i < getEdgeFromCurrSize(v1);++i)
  {
    h_vert_edge_from_data[v2*EDGE_DATA_BATCH_SIZE+getEdgeFromCurrSize(v2)] = getEdgeFromDataId(v1,i);

    //Update <from> pointer
    //cerr << "Updating from " << getEdgeFromDataId(v1,i) << endl;
    h_edges[EDGE_SIZE*getEdgeFromDataId(v1,i)] = v2;
    edgeFromIncreaseSize(v2);
  }

  for(int i = 0; i < getEdgeToCurrSize(v1); ++i)
  {
    h_vert_edge_to_data[v2*EDGE_DATA_BATCH_SIZE+getEdgeToCurrSize(v2)] = h_vert_edge_to_data[v1*EDGE_DATA_BATCH_SIZE+i];

    //Update <to> pointer
    h_edges[EDGE_SIZE*getEdgeToDataId(v1,i)+1] = v2;
    //cerr << "Updating to " << getEdgeToDataId(v1,i) << " >>> " << getEdgeVertexId(getEdgeToDataId(v1,i),0) << "-"<<getEdgeVertexId(getEdgeToDataId(v1,i),1) << endl;

    edgeToIncreaseSize(v2);
  }

  cerr << "v1 from ";
  for(int i = 0; i < getEdgeFromCurrSize(v1); ++i) cerr << getEdgeFromDataId(v1,i) << " ";
  cerr << "\nv1 to ";
  for(int i = 0; i < getEdgeToCurrSize(v1); ++i) cerr << getEdgeToDataId(v1,i) << " ";

  cerr << "\nv2 from ";
  for(int i = 0; i < getEdgeFromCurrSize(v2); ++i) cerr << getEdgeFromDataId(v2,i) << " ";
  cerr << "\nv2 to ";
  for(int i = 0; i < getEdgeToCurrSize(v2); ++i) cerr << getEdgeToDataId(v2,i) << " ";
  cerr << endl;

  //REMOVEEDGE ENDS

  //REMOVE VERTEX
  h_vertex_removed[v1] = true;

  cerr << "Finished collapse\n";

}

void updateEdgeCosts(int vid)
{
  cerr <<"Updating costs for vid " << vid << endl;
  for(int i = 0 ; i < getEdgeFromCurrSize(vid); ++i)
  {
    int eid = getEdgeFromDataId(vid,i);
    if(isEntirelyInCell(eid) && isCrownInCell(getEdgeVertexId(eid,0)) && isCrownInCell(getEdgeVertexId(eid,1)))
    {
      h_edge_cost[eid] = getCost(eid);
    }
  }
  for(int i = 0; i < getEdgeToCurrSize(vid); ++i)
  {

    int eid = getEdgeToDataId(vid,i);
    cerr << "Updating eid " << eid << " " << h_edges[eid*EDGE_SIZE] << "-" << h_edges[eid*EDGE_SIZE+1] << " = " << h_edge_cost[eid] << " ";
    if(isEntirelyInCell(eid) && isCrownInCell(getEdgeVertexId(eid,0)) && isCrownInCell(getEdgeVertexId(eid,1)))
    {
      h_edge_cost[eid] = getCost(eid);
    }
    cerr << " = " << h_edge_cost[eid] << endl;
  }

  //std::sort(h_cell_queue.data()+(h_vertex_in_cell[vid]*n_cells),h_cell_queue.data()+(h_vertex_in_cell[vid]*n_cells+h_cell_queue_size[vid]),compareEdges);
}

void SimpGPU::simplify(int goal, int gridres=1)
{
  grid_res = gridres;
  cerr << "Initializing Data Structures...\n";
  initDataStructures();
  cerr << "Computing initial quadrics...\n";
  initQuadrics();
  cerr << "Computing edges...\n";
  initEdges();
  cerr << "Initializing Uniform Grid...\n";


  int vertices_removed = 0;
  cerr << "Target vertex count: " << n_vertices - goal << endl;

  //STILL NEED TO_EDGES

  //while(vertices_removed < goal)
  {
    initUniformGrid();
    cerr << greentty << "Grid: " << grid_res << deftty << endl;

    //Simplify each cell
    for(int i = 0; i < n_cells; ++i)
    {
      int vr = 0;
      if(h_cell_queue_size[i] == 0) continue; //Cell is empty, move to next one

      while(vr < h_initial_vertices[i]/grid_res && vertices_removed < goal && h_cell_queue_size[i]>0)
      {
        int queue_it = h_cell_queue_size[i]-1;
        cerr << "cell " << i << " queue_it " << queue_it << endl;
        //Last position should contain the least cost edge
        int eid = h_cell_queue[i*n_cells+queue_it];
        h_cell_queue_size[i]--;


        collapse(eid);
        quadricAdd(h_quadrics.data()+(h_edges[eid*EDGE_SIZE+1]*QUADRIC_SIZE), h_quadrics.data()+(h_edges[eid*EDGE_SIZE]*QUADRIC_SIZE));
        vr++;
        vertices_removed++;
        updateEdgeCosts(getEdgeVertexId(eid,1));
        std::sort(h_cell_queue.data()+(i*n_cells),h_cell_queue.data()+(i*n_cells+h_cell_queue_size[i]),compareEdges);
      }
    }
  }


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
  h_vert_edge_from_header.resize(HEADER_SIZE*n_vertices);
  h_vert_edge_from_data.resize(EDGE_DATA_BATCH_SIZE*n_vertices);
  h_vert_edge_to_header.resize(HEADER_SIZE*n_vertices);
  h_vert_edge_to_data.resize(EDGE_DATA_BATCH_SIZE*n_vertices);

  //init HEADER array
  for(int i = 0; i < n_vertices; ++i)
  {
    h_vert_edge_from_header[HEADER_SIZE*i] = EDGE_DATA_BATCH_SIZE*i;
    h_vert_edge_from_header[HEADER_SIZE*i+1] = 0;
    h_vert_edge_from_header[HEADER_SIZE*i+2] = -1;

    h_vert_edge_to_header[HEADER_SIZE*i] = EDGE_DATA_BATCH_SIZE*i;
    h_vert_edge_to_header[HEADER_SIZE*i+1] = 0;
    h_vert_edge_to_header[HEADER_SIZE*i+2] = -1;
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
      //cerr << "eid: " << eid << " | " << vid[w[i][0]] << "," << vid[w[i][1]] << endl;

      for(int j = 0; j < getEdgeFromCurrSize(vid[w[i][0]]); ++j)
      {
        //Check wheter edge already exists
        if(getEdgeVertexId(getEdgeFromDataId(vid[w[i][0]],j),1) == vid[w[i][1]])
          found_edge = true;
      }
      if(!found_edge)
      {
        h_edges[EDGE_SIZE*eid] = vid[w[i][0]];
        h_edges[EDGE_SIZE*eid+1] = vid[w[i][1]];
        //cerr << "adding " << eid << " to " << getEdgeFromHeaderPos(vid[w[i][0]])+getEdgeFromCurrSize(vid[w[i][0]]) << endl;
        h_vert_edge_from_data[getEdgeFromHeaderPos(vid[w[i][0]])+getEdgeFromCurrSize(vid[w[i][0]])] = eid;
        //cerr << "added: " << h_vert_edge_data[getEdgeFromHeaderPos(vid[w[i][0]])+getEdgeFromCurrSize(vid[w[i][0]])] << endl;
        //h_vert_edge_data[getEdgeFromHeaderPos(vid[w[i][1])+getEdgeFromCurrSize(vid[w[i][1]])]]=eid;
        edgeFromIncreaseSize(vid[w[i][0]]);
        //edgeFromIncreaseSize(vid[w[i][1]]);

        //Add edge_to
        h_vert_edge_to_data[getEdgeToHeaderPos(vid[w[i][1]])+getEdgeToCurrSize(vid[w[i][1]])] = eid;
        edgeToIncreaseSize(vid[w[i][1]]);
        eid++;
      }
    }


  }

  //COMPUTE COSTS
  n_edges = eid;
  h_edge_cost.resize(n_edges);
  h_edge_removed.resize(n_edges);
  h_edge_queue.resize(n_edges);
  for(int i = 0; i < eid; ++i)
  {
    h_edge_cost[i] = getCost(i);
    h_edge_queue[i] = i;
    //cerr << "edge " << i << " - cost " << h_edge_cost[i] << " - removed " << h_edge_removed[i] << endl;
  }

  std::sort(h_edge_queue.data(),h_edge_queue.data()+n_edges,compareEdges);

  gettime(te1);
  te = diff(te0,te1);
  cerr << "Time to init edges: " << getMilliseconds(te) << endl;
  cerr << "No. of edges: " << n_edges <<endl;

  for(int i = 0; i < n_edges; ++i)
  {
    cerr << "edge " << i << ": " << h_edges[EDGE_SIZE*i] << " - " << h_edges[EDGE_SIZE*i+1] << endl;
  }
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
