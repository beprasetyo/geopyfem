// Gmsh project created on Sat Sep 26 10:52:31 2026
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {1, 0, 0, 1.0};
//+
Point(3) = {1, 1, 0, 1.0};
//+
Point(4) = {0, 1, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Curve Loop(1) = {4, 1, 2, 3};
//+
Plane Surface(1) = {1};
//+
Transfinite Curve {4} = 10 Using Progression 1;
//+
Transfinite Curve {1} = 10 Using Progression 1;
//+
Transfinite Curve {2} = 10 Using Progression 1;
//+
Transfinite Curve {3} = 10 Using Progression 1;
//+
Transfinite Surface {1};
//+
Recombine Surface {1};
//+
Physical Curve("fixed", 5) = {4};
//+
Physical Curve("load", 6) = {2};
//+
Physical Surface("soil1", 7) = {1};
//+
Physical Point("titik", 8) = {1};
