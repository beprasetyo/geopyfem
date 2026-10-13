// Gmsh project created on Thu Sep 24 20:39:38 2026
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {1, 0, 0, 1.0};
//+
Point(3) = {1, 2, 0, 1.0};
//+
Point(4) = {0, 2, 0, 1.0};

//+
Point(5) = {1, 1, 0, 1.0};
//+
Point(6) = {0, 1, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 5};
//+
Line(3) = {5, 6};
//+
Line(4) = {6, 1};
//+
Line(5) = {6, 4};
//+
Line(6) = {4, 3};
//+
Line(7) = {3, 5};
//+
Curve Loop(1) = {4, 1, 2, 3};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {5, 6, 7, 3};
//+
Plane Surface(2) = {2};
//+
Physical Curve("bottomfix", 8) = {1};
//+
Physical Curve("load", 9) = {6};
//+
Physical Surface("soil1", 10) = {2};
//+
Physical Surface("soil2", 11) = {1};
//+
Transfinite Curve {4} = 10 Using Progression 1;
//+
Transfinite Curve {5} = 10 Using Progression 1;
//+
Transfinite Curve {2} = 10 Using Progression 1;
//+
Transfinite Curve {7} = 10 Using Progression 1;
//+
Transfinite Curve {1} = 10 Using Progression 1;
//+
Transfinite Curve {3} = 10 Using Progression 1;
//+
Transfinite Curve {6} = 10 Using Progression 1;

//+
Transfinite Surface {2};
//+
Transfinite Surface {1};

//+
Physical Curve("sidefix", 12) = {5, 4, 7, 2};
