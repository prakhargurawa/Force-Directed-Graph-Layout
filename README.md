# Force-Directed-Graph-Layout

This is an interesting application of Gradient descent. Although we are not using gradient here but it exploits the concepts of gradient descent to stabalize a graph containing multiple nodes.
Each node expreience two kinds of repulsive forces:
1. Spring forces (between any two connected nodes)
2. Magnetic forces (between non-connected nodes) also known as Inverse square law.

Although the distance between any two node is fixed (by default equal to one) which keep the structure intact.

These is how these forces act and leads to a stable graph:

![alt text](https://github.com/prakhargurawa/Force-Directed-Graph-Layout/blob/main/GIF/grid_graph_gif.gif?raw=true)
![alt text](https://github.com/prakhargurawa/Force-Directed-Graph-Layout/blob/main/GIF/grid_graph_3D_gif.gif?raw=true)
