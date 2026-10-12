// Gmsh project created on Thu Oct  1 21:09:59 2026
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {10, 0, 0, 1.0};
//+
Point(3) = {10, 4, 0, 1.0};
//+
Point(4) = {0, 4, 0, 1.0};
//+
Point(5) = {10, 1, 0, 1.0};
//+
Point(6) = {0, 1, 0, 1.0};
//+
Point(7) = {10, 2, 0, 1.0};
//+
Point(8) = {10, 3, 0, 1.0};
//+
Point(9) = {0, 2, 0, 1.0};
//+
Point(10) = {0, 3, 0, 1.0};
//+
Point(11) = {8, 4, 0, 1.0};
//+
Point(12) = {6, 3, 0, 1.0};
//+
Point(13) = {4, 2, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 5};
//+
Line(3) = {5, 6};
//+
Line(4) = {6, 1};
//+
Line(5) = {6, 9};
//+
Line(6) = {9, 13};
//+
Line(7) = {13, 7};
//+
Line(8) = {7, 5};
//+
Line(9) = {7, 8};
//+
Line(10) = {8, 12};
//+
Line(11) = {12, 10};
//+
Line(12) = {10, 9};
//+
Line(13) = {13, 12};
//+
Line(14) = {8, 3};
//+
Line(15) = {3, 11};
//+
Line(16) = {11, 4};
//+
Line(17) = {4, 10};
//+
Line(18) = {12, 11};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {3, 5, 6, 7, 8};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {6, 13, 11, 12};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {7, 9, 10, -13};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {10, 18, -15, -14};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {11, -17, -16, -18};
//+
Plane Surface(6) = {6};
//+
Physical Curve("bottomfix", 19) = {1};
//+
Physical Curve("sidefix", 20) = {2, 8, 9, 14, 4, 5, 12, 17};
//+
Physical Curve("load", 21) = {15};
//+
Physical Surface("soil1", 22) = {2};
//+
Physical Surface("soil2", 23) = {1};
//+
Physical Surface("soil3", 24) = {4};
//+
Physical Surface("soil4", 25) = {5};
//+
Physical Surface("excav_soil4", 26) = {6};
//+
Physical Surface("excav_soil3", 27) = {3};
//+
Transfinite Curve {17, 12, 5, 4, 14, 9, 8, 2} = 5 Using Progression 1;
//+
Transfinite Curve {1, 3} = 30 Using Progression 1;
//+
Transfinite Curve {6, 10} = 15 Using Progression 1;
//+
Transfinite Curve {11, 7} = 20 Using Progression 1;
//+
Transfinite Curve {15} = 5 Using Progression 1;
//+
Transfinite Curve {16} = 25 Using Progression 1;
//+
Transfinite Curve {13, 18} = 6 Using Progression 1;
