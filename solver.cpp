#include "headers.hpp"

extern int myRank;
extern int nbTasks;

//================================================================================
// Solution of the system Au=b with Jacobi
//================================================================================
double prod(Vector& u, Vector& v, Mesh& mesh)
{
  int size = u.rows();
  double produit=0;
  for(int i=0; i<size; i++){ produit+=u(i)*v(i)/mesh.countShareNodes(i);}
  MPI_Allreduce(&produit, &produit, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return produit;
}

void jacobi(SpMatrix& A, Vector& b, Vector& u, Mesh& mesh, double tol, int maxit)
{
  if(myRank == 0)
    printf("== jacobi\n");

  // Compute the solver matrices
  int size = A.rows();
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
  exchangeAddInterfMPI(Mdiag, mesh);
  
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

    // Update residual
    Vector Au = A*u;
    exchangeAddInterfMPI(Au, mesh);
    Vector residu = Au-b;
    double residuNormTmp = 0;

    for(int noeud=0; noeud<size; noeud++){
      residuNormTmp = residuNormTmp + abs(pow(residu(noeud),2))/mesh.countShareNodes(noeud);  // The nodes in the two domains are counted twice
    }

    MPI_Allreduce(&residuNormTmp, &residuNorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    residuNorm = sqrt(residuNorm);
    
    // Update iterator
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

  // Heure a la fin de la simulation
  MPI_Barrier(MPI_COMM_WORLD);

}

void gradient_conjugue(SpMatrix& A, Vector& b, Vector& u, Mesh& mesh, double tol, int maxit)
{
  if(myRank == 0){
    printf("== gradient_conjugue\n");}

  // Conjugate gradient solver  
  int size = A.rows();
  Vector r(size);
  double r_norm = 1;
  double alfa = 0;
  double beta = 0;
  Vector p(size);

  Vector Au=A*u;
  exchangeAddInterfMPI(Au, mesh);
  r = b-Au;
  p=r;

  r_norm=prod(r,r,mesh);  

  double tol_r=r_norm*tol;
  int it = 0;
  while (r_norm > tol_r && it < maxit){
    //Compute step    
    Vector Ap=A*p;
    exchangeAddInterfMPI(Ap, mesh);
    alfa=prod(r,p,mesh)/prod(Ap,p,mesh);

    //Update
    u=u+alfa*p;

    //Compute residual
    r=r-alfa*Ap;

    //Compute direction
    Vector Ar=A*r;
    exchangeAddInterfMPI(Ar, mesh);
    beta=-prod(Ar,p,mesh)/prod(Ap,p,mesh);
    p=r+beta*p;

    //Residual norm
    r_norm=sqrt(prod(r,r,mesh));

    // Update residual and iterator
    if((it % 10) == 0)
    {   
      if(myRank == 0) printf("\r   %i %e", it, r_norm);
    }
    it++;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if(myRank == 0){
    printf("\r   -> final iteration: %i (prescribed max: %i)\n", it, maxit);
    printf("   -> final residual: %e (prescribed tol: %e)\n", r_norm, tol_r);
  }

}
