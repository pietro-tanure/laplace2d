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
    uExa(i) = cos(2*M_PI*x)*cos(3*M_PI*y);
    f(i) = (1+13*M_PI*M_PI)*cos(2*M_PI*x)*cos(3*M_PI*y);
  }
  
  Problem pbm;
  double alpha = 1;
  buildProblem(pbm,mesh,alpha,f);
  

  // 4. Solve problem + Solver time
  double timeInit; double timeEnd;
  if(myRank==0){
    timeInit = MPI_Wtime();  // Heure debut simulation
    cout<< "timeInit: "<< timeInit << endl;
  }

  double tol = 0.000001; // (Currently useless)
  int maxit = 1 *pow(10,5);
  //jacobi(pbm.A, pbm.b, uNum, mesh, tol, maxit);
  gradient_conjugue(pbm.A, pbm.b, uNum, mesh, tol, maxit);
  
  MPI_Barrier(MPI_COMM_WORLD);
  if (myRank==0){
    timeEnd = MPI_Wtime();
    cout<< "timeEnd: "<< timeEnd << endl;
    cout<<"Temps de calcul: "<< timeEnd-timeInit << endl;
  }


  // 5. Compute error and export fields
  Vector uErr = uNum - uExa;
  double errNorm; double NormuExa;
  double errNormTmp = uErr.transpose()*pbm.M*uErr;
  double NormuExaTmp = uExa.transpose()*pbm.M*uExa;
  MPI_Allreduce(&errNormTmp, &errNorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&NormuExaTmp, &NormuExa, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  errNorm = sqrt(errNorm)/sqrt(NormuExa);
  if(myRank == 0) printf("Norm d'erreur L2: %e \n",errNorm);
  saveToMsh(uNum, mesh, "solNum", "benchmark/solNum.msh");
  saveToMsh(uExa, mesh, "solRef", "benchmark/solExa.msh");
  saveToMsh(uErr, mesh, "solErr", "benchmark/solErr.msh");  

  
  // 6. Finilize MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  
  return 0;
}
