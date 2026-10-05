// Gmsh project created on Thu Oct  1 14:16:51 2026
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {10, 0, 0, 1.0};
//+
Point(3) = {10, 4, 0, 1.0};
//+
Point(4) = {10, 2, 0, 1.0};
//+
Point(5) = {0, 2, 0, 1.0};
//+
Point(6) = {10, 1, 0, 1.0};
//+
Point(7) = {10, 3, 0, 1.0};
//+
Point(8) = {8, 4, 0, 1.0};
//+
Point(9) = {6, 3, 0, 1.0};
//+
Point(10) = {4, 2, 0, 1.0};
//+
Point(11) = {0, 1, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 6};
//+
Line(3) = {6, 11};
//+
Line(4) = {11, 1};
//+
Line(5) = {11, 5};
//+
Line(6) = {5, 10};
//+
Line(7) = {10, 4};
//+
Line(8) = {4, 7};
//+
Line(9) = {7, 9};
//+
Line(10) = {9, 10};
//+
Line(11) = {9, 8};
//+
Line(12) = {8, 3};
//+
Line(13) = {3, 7};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Line(14) = {4, 6};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {5, 6, 7, 14, 3};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {7, 8, 9, 10};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {13, 9, 11, 12};
//+
Plane Surface(4) = {4};
//+
Physical Curve("bottomfix", 15) = {1};
//+
Physical Curve("sidefix", 16) = {5, 4, 2, 14, 8, 13};
//+
Physical Curve("load", 17) = {12};
//+
Physical Surface("soil1", 18) = {2};
//+
Physical Surface("soil2", 19) = {1};
//+
Physical Surface("timbunan_bawah", 20) = {3};
//+
Transfinite Curve {5, 4, 14, 2, 8, 13} = 5 Using Progression 1;
//+
Transfinite Curve {1, 3} = 30 Using Progression 1;
//+
Transfinite Curve {6, 7} = 20 Using Progression 1;
//+
Transfinite Curve {9} = 20 Using Progression 1;
//+
Transfinite Curve {12} = 10 Using Progression 1;
//+
Transfinite Curve {10, 11} = 10 Using Progression 1;
//+
Transfinite Surface {1};
//+
Transfinite Surface {2};
//+
Transfinite Surface {3};
//+
Transfinite Surface {4};

//+
Physical Surface("timbunan_atas", 21) = {4};
