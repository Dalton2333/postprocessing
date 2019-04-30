from abaqus import *
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *

# import os
# para_path = os.getcwd()+'/'
# para_path = void_dir_path + '/'
# sys.path.insert(0, para_path)
import parameters

ap = parameters.ap


model_name = 'Model-1'


model1 = mdb.Model(name = model_name)
viewport1 = session.Viewport(name='viewport_1', origin=(20,20), width=150, height = 120)
sketch1 = model1.ConstrainedSketch(name='__profile__', sheetSize=200.0)

import json
# ------------- need to convert (x,z) to (z,x)
with open(ap['test_dir_path'] + 'outside_coords.txt') as file:
    outside_coords = json.load(file)
    # outside_coords = ast.literal_eval(outside_coords)
for i in range(len(outside_coords)-1):
    sketch1.Line(point1=(outside_coords[i][1],outside_coords[i][0]),
                 point2=(outside_coords[i+1][1],outside_coords[i+1][0]))
sketch1.Line(point1=(outside_coords[-1][1],outside_coords[-1][0]),
             point2=(outside_coords[0][1],outside_coords[0][0]))
with open(ap['test_dir_path'] + 'circles.txt') as file:
    circles = json.load(file)
for circle in circles:
    sketch1.CircleByCenterPerimeter(center=(circle[0][1], circle[0][0]),
                                    point1=(circle[0][1]+circle[1], circle[0][0]))

part1 = model1.Part(dimensionality=TWO_D_PLANAR, name='Part-1', type=
    DEFORMABLE_BODY)
part1.BaseShell(sketch=sketch1)
