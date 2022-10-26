#include "headers.hpp"

int myRank;
int nbTasks;

int main(int argc, char* argv[])
{
  
  // 1. Initialize MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nbTasks);
 
  // 2. Read the mesh, and build lists of nodes for MPI exchanges, local numbering
  Mesh mesh;
  readMsh(mesh, "benchmark/mesh.msh");
  buildListsNodesMPI(mesh);
  
  // 3. Build problem (vectors and matrices)
  Vector uNum(mesh.nbOfNodes);
  Vector uExa(mesh.nbOfNodes);
  Vector f(mesh.nbOfNodes);
  for(int i=0; i<mesh.nbOfNodes; ++i){
    double x = mesh.coords(i,0);
    double y = mesh.coords(i,1);
    uNum(i) = 0.;
    uExa(i) = x+y;
    f(i) = x+y;
  }
  
  Problem pbm;
  double alpha = 1;
  buildProblem(pbm,mesh,alpha,f);
  
  // 4. Solve problem
  double tol = 1e-9; // (Currently useless)
  int maxit = 1000;
  jacobi(pbm.A, pbm.b, uNum, mesh, tol, maxit);
  
  // 5. Compute error and export fields
  Vector uErr = uNum - uExa;
  saveToMsh(uNum, mesh, "solNum", "benchmark/solNum.msh");
  saveToMsh(uExa, mesh, "solRef", "benchmark/solExa.msh");
  saveToMsh(uErr, mesh, "solErr", "benchmark/solErr.msh");

  // 6. Finilize MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  
  return 0;
}
