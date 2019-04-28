''' this file serves as a case to implement the whole process, from inp to both-inner-and-outer boundary, to
sorted boundary to find straight line, to smooth the zigzag boundary parts, to connect all of them and plot

def Get_all_boundary_nodes_from_inp(fh, both_boundary_nodes):
def find_outside_boundary(all_nodes, outside):
def find_x_min(onelist,x_min_Formal_parameter):
def sort_boundary_from_x_min(onelist,x_min_Formal_parameter,Newlist):
def find_straight_line(listnode,threshold_number,emptylist):

def B(x, p, i, t):
def bspline_x(x, t, c, p):
def bspline_y(x, t, c, p):
def solve_knot_vector(t,p,n,start_de,end_de,knotvector):
def smooth_and_get_chord_length(input_coarse,output_chordlength):
def Get_Control(input_coarse, chord_length, p, t, n, outputControlPoint):
def obtain_xy_of_bspline(input_coarse,output_xy_boundary,Control_list,n,degree=3):

def merge_straght_line(s_to_e,new_s_to_e):
def smooth_boundary_by_connecting(outside, s_to_e,new_outside):

'''
import all_function as kz
import matplotlib.pyplot as plt
import math
import numpy
import parameters


fh = open(parameters.inp_path)
#fh = open('Cantilever2-99.inp')
# fh=open('Case_SimplySupportedBeam2-40.inp')
#fh=open('Michell1-49.inp')

both_nodes=[]
kz.Get_both_boundary_nodes_from_inp(fh,both_nodes)
outside=[]
kz.find_outside_boundary(both_nodes,outside)
#--------------------------------plot both-side boundary
plot_outside=outside
both_nodes = [tuple(l) for l in both_nodes]
#print(outside)
tr_both=list(numpy.transpose(both_nodes))
fig,ax=plt.subplots()
plt.scatter(tr_both[0],tr_both[1],s=2,c="red")
plt.title('inner and outer boundary')

# -----------------plot the outside figure to check it
tr_outside=list(numpy.transpose(outside))
fig,ax=plt.subplots()
plt.scatter(tr_outside[0],tr_outside[1],s=2, c="red")
plt.plot(tr_outside[0],tr_outside[1])
plt.title('outside boundary')
#-----------------------------------------

# -----------------plot the inside boundary
##tr_inside = list(set(tr_both)-set(tr_outside))
#inside = list(set(both_nodes)-set(outside))
#tr_inside = list(numpy.transpose(inside))
#fig, ax = plt.subplots()
#plt.scatter(tr_inside[0],tr_inside[1],s=1.75)
##plt.plot(tr_outside[0],tr_outside[1])
#plt.title('inside boundary')
#-----------------------------------------

x_min_list=[]
kz.find_x_min(outside,x_min_list)
Sorted_newlist=[]
#print(x_min_list,'line104')
kz.sort_boundary_from_x_min(outside,x_min_list,Sorted_newlist)
#print(len(Sorted_newlist),len(outside),Sorted_newlist)

outside=Sorted_newlist
ts_temp=list(numpy.transpose(outside))
fig, ax = plt.subplots()
# ax.scatter(ts_coordinate_bspline[0],ts_coordinate_bspline[1],s=0.5,marker='o')
ax.plot(ts_temp[0],ts_temp[1],'r-',lw=0.75)
ax.scatter(ts_temp[0],ts_temp[1],s=2,marker='o')

ax.grid(True)
plt.title('original outside boundary')

list_of_line=[]
threshold_number=9
kz.find_straight_line(Sorted_newlist,threshold_number,list_of_line)
#print(list_of_line)

plot_line=[]
for i in range(len(list_of_line)):
    plot_line.extend(Sorted_newlist[list_of_line[i][0]:list_of_line[i][1]])
tr_plot_line=list(numpy.transpose(plot_line))
fig,ax=plt.subplots()
plt.scatter(tr_plot_line[0],tr_plot_line[1],s=0.75)
plt.title('threshold number is %d'%(threshold_number))

s_to_e=[]
kz.merge_straght_line(list_of_line,s_to_e)
newoutside=[]
#print('kzkzkz',outside)

kz.smooth_boundary_by_connecting(outside,s_to_e,newoutside)
# s_to_e: start to end, tuple of node number (start, end) for straight line

# outside=newoutside

ts_coordinate_bspline=list(numpy.transpose(newoutside))
tr_o = list(numpy.transpose(plot_outside))

#fig, ax = plt.subplots()
## ax.scatter(ts_coordinate_bspline[0],ts_coordinate_bspline[1],s=0.5,marker='o')
#ax.plot(ts_coordinate_bspline[0],ts_coordinate_bspline[1],'r-',lw=0.9,label='Bspline')
#ax.scatter(tr_o[0],tr_o[1],s=1,marker='o',label='original boundary')
#plt.title('overall outside boundary')
#ax.grid(True)
#plt.legend(loc="best")
#plt.show()

#-------The final plot
fig, ax = plt.subplots()
# ax.scatter(ts_coordinate_bspline[0],ts_coordinate_bspline[1],s=0.5,marker='o')
ax.plot(ts_coordinate_bspline[0],ts_coordinate_bspline[1],'r-',lw=2)
#ax.scatter(tr_both[0],tr_both[1],s=0.75)
#,label='External boundary'
#ax.scatter(tr_o[0],tr_o[1],s=1,marker='o',label='original boundary')
from matplotlib.patches import Circle
from matplotlib.patheffects import withStroke

# ---------circle
 #(51.5, 19.3), 1.784
# (68.3, 17.6), 1.693
if parameters.inp_path == parameters.dirname+r"Cantilever2-4.inp":
    cx = 68.3
    cy = 17.6
    cr = 1.693
if parameters.inp_path == parameters.dirname + r"LBracket-9.inp":
    cx = 52.7
    cy = 19.5
    cr = 1.784

circle=Circle((cx, cy), cr, clip_on=False, zorder=10, linewidth=2,
                edgecolor='red', facecolor='none', label='Internal void',
                path_effects=[withStroke(linewidth=1, foreground='w')])
ax.add_artist(circle)


plt.title('overall outside boundary')
ax.grid(False)
#plt.legend(loc="best")
plt.show()

# -------The spline plot
#fig, ax = plt.subplots()
## ax.scatter(ts_coordinate_bspline[0],ts_coordinate_bspline[1],s=0.5,marker='o')
#ax.plot(ts_coordinate_bspline[0],ts_coordinate_bspline[1],'r-',lw=2)
##ax.scatter(tr_both[0],tr_both[1],s=0.75)
##,label='External boundary'
##ax.scatter(tr_o[0],tr_o[1],s=1,marker='o',label='original boundary')

#plt.title('overall outside boundary')
#ax.grid(False)
#plt.show()