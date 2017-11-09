Tetrahedral pathtracing using Tetgen based on nodes
===================================================

This project is similar to the tetra_mesh project which uses a tetrahedral mesh as acceleration structure for pathtracing. Here, only the vertices/nodes of the scene are used for the tetrahedral mesh, and for each generated tetrahedral cell the indices of the surrounding faces are saved.

This allows the use of "defect" meshes, e.g. with self-intersections, as acceleration structure for path tracing.

The pathtracer code is based on the pathtracer by Samuel Lapere (https://github.com/straaljager/GPU-path-tracing-tutorial-3)
and the smallpt pathtracer (http://kevinbeason.com/smallpt/).

**References:**

Hang Si (2015). _TetGen, a Delaunay-Based Quality Tetrahedral Mesh Generator._ ACM Trans. on Mathematical Software. 41 (2), Article 11 (February 2015), 36 pages.

Lagae, A. and Dutré, P. (2008). _Accelerating Ray Tracing using Constrained Tetrahedralizations._ Computer Graphics Forum, 27: 1303–1312.

Sanzenbacher, S. (2010) Darstellung von Kaustiken und Lichtstreuung in Interaktiven Anwendungen. _Unpublished diploma thesis, Institut für Visualisierung und Interaktive Systeme, University Stuttgart._
  
**Current status (06/21/2017):**  

Loading wavefront .obj nodes - done  
Implement tetrahedralization via tetgen - done  
Associate face indices to nodes - done  
Load mesh into global memeory on GPU - done  
  
09.11.2017: With bigger scenes,  numerical errors appear, therefore this projects is abandoned and a  
a new approach tested.  


