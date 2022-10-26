#include "headers.hpp"

extern int myRank;
extern int nbTasks;

//================================================================================
// Compute the matrices of the linear system
//================================================================================

void buildProblem(Problem& pbm, Mesh& mesh, double alpha, Vector& f)
{
  if(myRank == 0)
    printf("== build linear system\n");
  
  pbm.K.resize(mesh.nbOfNodes, mesh.nbOfNodes);
  pbm.M.resize(mesh.nbOfNodes, mesh.nbOfNodes);
  
  for(int iTriLoc=0; iTriLoc<mesh.nbOfTri; iTriLoc++){
    
    IntVector s = mesh.triNodes.row(iTriLoc);
    Vector s0 = mesh.coords.row(s(0));
    Vector s1 = mesh.coords.row(s(1));
    Vector s2 = mesh.coords.row(s(2));
    
    double x01 = 0.5*(s1(0)-s0(0));
    double y01 = 0.5*(s1(1)-s0(1));
    double x12 = 0.5*(s2(0)-s1(0));
    double y12 = 0.5*(s2(1)-s1(1));
    double x20 = 0.5*(s0(0)-s2(0));
    double y20 = 0.5*(s0(1)-s2(1));
    double detJ = abs(x12*y20 - x20*y12);
    
    // Elemental mass matrix
    double Mel[3][3];
    for(int i=0; i<3; i++){
      for(int j=0; j<3; j++){
        Mel[i][j] = (i==j) ? 1./3. : 1./6.;
        Mel[i][j] *= detJ;
      }
    }
    
    // Elemental stiffness matrix
    double Kel[3][3];
    Kel[0][0] = (x12*x12+y12*y12)/(2.*detJ);
    Kel[0][1] = (x12*x20+y12*y20)/(2.*detJ);
    Kel[0][2] = (x12*x01+y12*y01)/(2.*detJ);
    Kel[1][0] = (x12*x20+y12*y20)/(2.*detJ);
    Kel[1][1] = (x20*x20+y20*y20)/(2.*detJ);
    Kel[1][2] = (x20*x01+y20*y01)/(2.*detJ);
    Kel[2][0] = (x12*x01+y12*y01)/(2.*detJ);
    Kel[2][1] = (x20*x01+y20*y01)/(2.*detJ);
    Kel[2][2] = (x01*x01+y01*y01)/(2.*detJ);
    
    // Assembling
    for(int i=0; i<3; ++i){
      for(int j=0; j<3; ++j){
        pbm.M.coeffRef(s(i),s(j)) += Mel[i][j];
        pbm.K.coeffRef(s(i),s(j)) += Kel[i][j];
      }
    }
  }
  
  pbm.A = alpha * pbm.M + pbm.K;
  pbm.b = pbm.M * f;
  exchangeAddInterfMPI(pbm.b, mesh);
}