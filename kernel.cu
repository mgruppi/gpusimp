#include "kernel.cuh"

//get position from header array
//Size of each structure on arrays
#define HEADER_SIZE  3
#define FACE_SIZE 3
#define VERTEX_SIZE 3
#define FACE_DATA_BATCH_SIZE 50
#define EDGE_DATA_BATCH_SIZE 50
#define QUADRIC_SIZE 16 //Quadric for a vertex is a 4x4 matrix
#define EDGE_SIZE 2

//UGRID
#define CELL_HEADER_SIZE 2

//ACCESS Vertex
#define getFaceVertexId(face,vertex) faces[FACE_SIZE*face+vertex]
#define getFaceHeaderPos(vertex) vert_face_header[HEADER_SIZE*vertex]
#define getFaceCurrSize(vertex) vert_face_header[HEADER_SIZE*vertex+1]
#define faceIncreaseSize(vertex) vert_face_header[HEADER_SIZE*vertex+1]++
#define faceDecreaseSize(vertex) vert_face_header[HEADER_SIZE*vertex+1]--
#define getFaceId(vid,p) vert_face_data[getFaceHeaderPos(vid)+p]
#define getX(vid) vertices[VERTEX_SIZE*vid]
#define getY(vid) vertices[VERTEX_SIZE*vid+1]
#define getZ(vid) vertices[VERTEX_SIZE*vid+2]

//ACCESS TO EDGES
#define getEdgeVertexId(edge,vid) edges[EDGE_SIZE*edge+vid] //Get vertex id (0 or 1) of edge
#define getEdgeFromHeaderPos(vid) vert_edge_from_header[HEADER_SIZE*vid]
#define getEdgeFromCurrSize(vid) vert_edge_from_header[HEADER_SIZE*vid+1]
#define edgeFromIncreaseSize(vid) assert(vert_edge_from_header[HEADER_SIZE*vid+1] < EDGE_DATA_BATCH_SIZE); vert_edge_from_header[HEADER_SIZE*vid+1]++
#define edgeFromDecreaseSize(vid) vert_edge_from_header[HEADER_SIZE*vid+1]--
#define getEdgeFromDataId(vid,p) vert_edge_from_data[getEdgeFromHeaderPos(vid)+p]

//edge_to
#define edgeToIncreaseSize(vid) assert(vert_edge_from_header[HEADER_SIZE*vid+1] < EDGE_DATA_BATCH_SIZE); vert_edge_to_header[HEADER_SIZE*vid+1]++
#define edgeToDecreaseSize(vid) vert_edge_to_header[HEADER_SIZE*vid+1]--
#define getEdgeToHeaderPos(vid) vert_edge_to_header[HEADER_SIZE*vid]
#define getEdgeToCurrSize(vid) vert_edge_to_header[HEADER_SIZE*vid+1]
#define getEdgeToDataId(vid,p) vert_edge_to_data[getEdgeToHeaderPos(vid)+p]

//Access
#define getPlacementX(vid1,vid2) (getX(vid1)+getX(vid2))/2
#define getPlacementY(vid1,vid2) (getY(vid1)+getY(vid2))/2
#define getPlacementZ(vid1,vid2) (getZ(vid1)+getZ(vid2))/2

//UNIFORM GRID
#define getCellHeaderPos(cell) cell_header[CELL_HEADER_SIZE*cell]
#define getCellHeaderSize(cell) cell_header[CELL_HEADER_SIZE*cell+1]
#define increaseCellSize(cell) cell_header[CELL_HEADER_SIZE*cell+1]++
#define getCellVertexId(cell,p) cell_vertices[cell*n_vertices+p]
#define getHeapHead(cell) cell*(n_edges+1)


//======DEVICE (GLOBAL) VARIABLES======
//VERTICES
__device__ double* vertices;
__device__ bool* vertex_removed;
__device__ double* quadrics;

//FACES
__device__ int* faces;
__device__ int* vert_face_header;
__device__ int* vert_face_data;
__device__ bool* face_removed;

//EDGES
__device__ int* edges;
__device__ double* edge_cost;
__device__ int* edge_queue;
__device__ int* vert_edge_from_header;
__device__ int* vert_edge_from_data;
__device__ int* vert_edge_to_header;
__device__ int* vert_edge_to_data;
__device__ bool* edge_removed;

//UNIFORM GRID
__device__ int n_cells;

//VERTEX
__device__ int* cell_data;
__device__ int* cell_header;
__device__ int* vertex_in_cell;
__device__ int* initial_vertices;
__device__ int* cell_vertices;
__device__ int* cell_vertices_size;

//EDGE
__device__ int* cell_heap;
__device__ int* cell_heap_size;


//UNIFORM GRID DIM AND OFFSET
__device__ double dim[3];
__device__ double offset[3];
__device__ int grid_res;


//NO.
__device__ int n_edges;
__device__ int n_faces;
__device__ int n_vertices;

//=====



//=====HEAP FUNCTIONS ======

#define compare(a,b) (a<b)

//Move node up
__device__ void d_percUp(int i, int* heap)
{
  while(i/2 > 0)
  {
    if(compare(heap[i], heap[i/2]))
    {
      int aux = heap[i];
      heap[i] = heap[i/2];
      heap[i/2] = aux;
    }
    i/=2;
  }
}

//Insert element e into heap of size n
__device__ void d_insert(int e, int* heap, int& n)
{
  heap[n+1] = e;
  n++;
  d_percUp(n, heap);
}

//Find child of least cost
__device__ int d_minChild(int i, int* heap, int n)
{
  //Does not have right child
  if (i * 2 + 1 > n) return i*2;
  else{
    if(compare(heap[2*i],heap[2*i+1])) return 2*i;
    else return 2*i+1;
  }
}


//Move node down the heap
__device__ void d_percDown(int i, int* heap, int n)
{
  while(i*2 <= n)
  {
    int mc = d_minChild(i, heap, n);
    if(compare(heap[mc],heap[i]))
    {
      int tmp = heap[mc];
      heap[mc] = heap[i];
      heap[i] = tmp;
    }
     i = mc;
  }
}

//Pop element of least value
 __device__ int d_pop(int* heap, int& n)
{
  int ret = heap[1];
  heap[1] = heap[n];
  n--;
  d_percDown(1, heap, n);
  return ret;
}

//Turn vector into heap or update heap
__device__ void d_heapify(int* vec, int n)
{
  int i = n/2;
  while( i > 0)
  {
    d_percDown(i,vec,n);
    i--;
  }
}
//===END OF HEAP===



__device__ bool isEntirelyInCell(int eid, int* edges, int* vertex_in_cell)
{
  //Return if both endpoints are withing the same cell
  return vertex_in_cell[getEdgeVertexId(eid,0)] == vertex_in_cell[getEdgeVertexId(eid,1)];
}

__device__ bool isCrownInCell(int vid, int* edges, int* vert_edge_from_header, int* vert_edge_from_data, int* vert_edge_to_header, int* vert_edge_to_data, int* vertex_in_cell)
{
  //Test edges leaving (from) vid to see if their endpoints also lie within vid's cell
  for(int i = 0 ; i < getEdgeFromCurrSize(vid); ++i)
  {
    //Check edge i from vid
    int eid = getEdgeFromDataId(vid,i);
    int endp = getEdgeVertexId(eid,1);
    if(vertex_in_cell[endp] != vertex_in_cell[vid])
      return false;
  }

  //Test edges arriving (to) vid
  for(int i = 0 ; i < getEdgeToCurrSize(vid); ++i)
  {
    int eid = getEdgeToDataId(vid,i);
    int endp = getEdgeVertexId(eid,0);
    if(vertex_in_cell[endp] != vertex_in_cell[vid])
      return false;
  }
  return true;
}



//environmentList
#define environmentList double* d_vertices, int d_n_vertices, bool* d_vertex_removed, double* d_quadrics, int* d_faces, int* d_vert_face_header, int* d_vert_face_data, bool* d_face_removed, int* d_edges, double* d_edge_cost, int* d_edge_queue, int* d_vert_edge_from_header, int* d_vert_edge_from_data, int* d_vert_edge_to_header, int* d_vert_edge_to_data, bool* d_edge_removed, int d_n_cells, int* d_vertex_in_cell, int* d_initial_vertices, int* d_cell_vertices, int* d_cell_vertices_size, int* d_cell_heap, int* d_cell_heap_size



//Initialize device environment
__global__ void initEnvironment(environmentList)
{
  vertices = d_vertices;
  n_vertices = d_n_vertices;
  vertex_removed = d_vertex_removed;
  quadrics = d_quadrics;
  faces = d_faces;
  vert_face_header = d_vert_face_header;
  vert_face_data = d_vert_face_data;
  face_removed = d_face_removed;
  edges = d_edges;
  vert_edge_from_header = d_vert_edge_from_header;
  vert_edge_from_data = d_vert_edge_from_data;
  vert_edge_to_header = d_vert_edge_to_header;
  vert_edge_to_data = d_vert_edge_to_data;
  edge_removed = d_edge_removed;
  n_cells = d_n_cells;
  vertex_in_cell = d_vertex_in_cell;
  initial_vertices = d_initial_vertices;
  cell_vertices = d_cell_vertices;
  cell_vertices_size = d_cell_vertices_size;
  cell_heap = d_cell_heap;
  cell_heap_size = d_cell_heap_size;
}

__global__ void addEdgesIntoUniformGrid(int* vertices, int n_vertices, int* edges, int* vert_edge_from_header,int* vert_edge_from_data, int* vert_edge_to_header, int* vert_edge_to_data, int* vertex_in_cell, int* cell_vertices, int* cell_vertices_size, bool* vertex_removed, int n_cells)
{
  int id = threadIdx.x;
  int size = n_cells/blockDim.x;

  for(int i = 0; i < size; ++i)
  {
    int cell = id*size+i;
    for(int j = 0;  j < cell_vertices_size[cell]; ++j)
    {
      for(int k = 0; k < getEdgeFromCurrSize(getCellVertexId(cell,j)); ++k)
      {

        int eid = getEdgeFromDataId(getCellVertexId(cell,j),k);
        if(isEntirelyInCell(eid, edges, vertex_in_cell) && isCrownInCell(getEdgeVertexId(eid,0), edges, vert_edge_from_header, vert_edge_from_data, vert_edge_to_header, vert_edge_to_data, vertex_in_cell) && isCrownInCell(getEdgeVertexId(eid,1),edges, vert_edge_from_header, vert_edge_from_data, vert_edge_to_header, vert_edge_to_data, vertex_in_cell))
        {
          //insert(eid, h_cell_heap.data()+getHeapHead(i), h_cell_heap_size[i]);
        }

      }
    }

  }

}


__device__ int* gg;

__global__ void initArray(int* d_a)
{
  gg = d_a;
}

__global__ void doubleArray(int n)
{
  //int size = n/blockDim.x;
  int id = threadIdx.x;

  if(id < n)
    gg[id] = 2*gg[id];

}

void initDeviceEnvironment(environmentList)
{
  
}

void allocate(int* h, int*& d_a)
{
  cudaMalloc(&d_a,100);

  cudaMemcpy(d_a, h, 100, cudaMemcpyHostToDevice);

  initArray<<<1,1>>>(d_a);

  doubleArray<<<1,100>>>(100);

  cudaMemcpy(h, d_a, 100, cudaMemcpyDeviceToHost);

  cudaFree(d_a);

}

void freeDevice(int* d_a)
{
  cudaFree(d_a);
}
