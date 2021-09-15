# Graph-Colouring-Using-ML
Trying out an ML technique to see if it can work for the problem of graph colouring.

## Procedure
1. Generate k-partite graphs for training.
    1. Each partition roughly has (n / k) nodes.
    2. Between any two nodes belonging to different vertices, edges are added with probability __p__.
It is crucial that we ensure that 'k' is the lowest number for which we can call the graph k-partite.
2. Splits.
    1. Test Train Split.
    2. All examples to be used only for train, testing to be done regular graphs.
3. ...

## Operations
1. Merge or 'L': Merge two non-adjacent vertices and form a new vertex. Edges connecting to either of the vertices will now connect to the new vertex. This can be understood as having coloured the vertices with the same colour.
2. Add Edge or 'R': Add an edge between 2 vertices. This can be understood as having coloured the vertices with different colours.

## Training
1. Use set of neighbourhood features to decide which operation to do, 'L' or 'R'.
2. The k-partite nature of the graph supplies labels for the two actions.
    1. For an 'L' type of action the labels are:
        1. 1 if chosen vertices are from the same partition.
        2. 0 if chosen vertices are from different partitions.
    2. For an 'R' type of action the labels are:
        1. 0 if chosen vertices are from the same partition.
        2. 1 if chosen vertices are from different partitions.

## Neighbourhood Diversity