
# **read me** 


## this package contains the following modules:
+ ### **mhxOpenGL.py** : a module to simplify the usage of OpenGL, 
	+ you only need to define draw(), <br/>and then plug into _mhxOpenGL.showUp(draw=draw)_
+ Arrow3D_useMhxOpenGL.py : an example for how to use module **mhxOpenGL.py**
+ ### **Element.py** : _a module including the following conponents_:
	+ class **C2D4** : quadrangle element with 4 nodes
		+ function **getPatch(n)** : a function to make patches inside the quadrangle, i.e. densify the grid
	+ class **C3D8** : _define C3D8 type element_
	+ class **object3D** : _use many C3D8 elements to construct a 3D object_
		+ function **getFaceEdge( )** _can automatically get the surfaces with corresponding edges of this object_
+ Arrow3D.py : _draw the arrow with function 'show'_
+ coordinateLine.py : _draw the coordinateLine with function 'draw( )'_
+ colorBar.py : _use different mods of color bar to get the color,  with function_ **'getColor( )'**
<br/><br/>
+ ./Abaqus/localDense.py : can locally denser the element and draw the 3D graph, also apple module **'mhxOpenGL'**
<br/><br/>
## some examples

+ folder _**'dataFile'**_ saves the data files that need to be read by python Scripts

+ **vectorField.py** 
	+ read the inp file to construct a 3D object, and draw it, <br/>also read the gradient file to draw the vector field (for example, the surface normal of twin)

+ readSurface.py : can ead the element surface file and draw the elements

+ getSurfaceEdge.py : can use the inp file of FEM to get all elements information and extract the surfaces as well as the surface edges

# **some further development**
+ eleNear 列表直接改为 ele.neighbor ? <br/>即把相邻单元的信息直接挂载到当前单元上？