//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {3, 0, 0, 1.0};
//+
Point(3) = {3, 6, 0, 1.0};
//+
Point(4) = {0, 6, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};
//+
Physical Curve("bottomfix", 5) = {1};
//+
Physical Curve("load", 6) = {3};
//+
Physical Surface("soil1", 7) = {1};
//+
Transfinite Curve {1, 3} = 5 Using Progression 1;
//+
Transfinite Curve {4, 2} = 9 Using Progression 1;
//+
Transfinite Surface {1};
//+
Recombine Surface {1};
