#include "headers.hpp"

extern int myRank;
extern int nbTasks;

//================================================================================
// Solution of the system Au=b with Jacobi
//================================================================================

//Ajouter une fonction pour le gradient conjugué

void jacobi(SpMatrix& A, Vector& b, Vector& u, Mesh& mesh, double tol, int maxit)
{
  if(myRank == 0)
    printf("== jacobi\n");
  
  // Compute the solver matrices
  int size = A.rows();  // Nombre de noeuds dans le sous-domaine
  Vector Mdiag(size);
  SpMatrix N(size, size);
  for(int k=0; k<A.outerSize(); ++k){
    for(SpMatrix::InnerIterator it(A,k); it; ++it){
      if(it.row() == it.col())
        Mdiag(it.row()) = it.value();
      else
        N.coeffRef(it.row(), it.col()) = -it.value();
    }
  }
  exchangeAddInterfMPI(Mdiag, mesh);  // Tous le paralléliser de l'algo
  
  // Jacobi solver
  double residuNorm = 1e2;
  int it = 0;
  while (residuNorm > tol && it < maxit){
    
    // Compute N*u
    Vector Nu = N*u;
    exchangeAddInterfMPI(Nu, mesh);
    
    // Update field
    for(int i=0; i<size; i++){
      u(i) = 1/Mdiag(i) * (Nu(i) + b(i));
    }
      
    // Update residual and iterator
    residuNorm = 0;
    for(int i=0; i<size; i++){
      residuNorm = residuNorm + abs(pow(u(i),2));
    }
    residuNorm = sqrt(residuNorm);

    if((it % 10) == 0){
      if(myRank == 0)
        printf("\r   %i %e", it, residuNorm);
    }
    it++;
  }
  
  if(myRank == 0){
    printf("\r   -> final iteration: %i (prescribed max: %i)\n", it, maxit);
    printf("   -> final residual: %e (prescribed tol: %e)\n", residuNorm, tol);
  }
}
