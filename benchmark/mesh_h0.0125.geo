Mesh.MshFileVersion = 2.2;

a = 1;
b = 1;
h = 0.0125; //h=0.05 

Point(1) = {0, 0, 0, h};
Point(2) = {a, 0, 0, h};
Point(3) = {a, b, 0, h};
Point(4) = {0, b, 0, h};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line Loop(1) = {1,2,3,4};

Plane Surface(1) = {1};
Physical Surface(1) = {1};
