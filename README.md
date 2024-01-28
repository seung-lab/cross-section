# xs3d: Compute cross sectional area for 3D image objects

```python
import xs3d

# let binary image be a boolean numpy array 
# in fortran order that is 500 x 500 x 500 voxels
# containing a shape, which may have multiple 
# connected components, representing e.g. a neuron
binary_image = np.load(...)

# a point inside the shape (must be integer)
vertex = np.array([200,121,78])
# normal vector defining sectioning plane
# it doesn't have to be a unit vector
# vector can be of arbitrary orientation
normal = np.array([0.01, 0.033, 0.9])

# voxel dimensions in e.g. nanometers
resolution = np.array([32,32,40]) 

# cross sectional area returned as a float
area = xs3d.cross_sectional_area(binary_image, vertex, normal, resolution)
```

When using skeletons (one dimensional stick figure representations) to create electrophysiological compartment simulations of neurons, some additional information is required for accuracy. The caliber of the neurite changes over the length of the cell.

Previously, the radius from the current skeleton vertex to the nearest background voxel was used, but this was often an underestimate as it is sensitive to noise and divots in a shape.

A superior measure would be the cross sectional area using a section plane that is orthogonal to the direction of travel along the neurite. This library provides that missing capability.

# How Does it Work?

The algorithm roughly works as follows.

1. Label voxels that are intercepted by the sectioning plane.
2. Label the connected components of those voxels.
3. Filter out all components except the region of interest.
4. Compute the polygon formed by the intersection of the plane with the 8 corners and 12 edges of each voxel.
5. Add up the area contributed by each polygon so formed in the component of interest.







