import re
import numpy as np
import xlrd
import xlwt
import math
import numpy

def B(x, p, i, t):
   if p == 0:
      return 1.0 if t[i] <= x < t[i+1] or ( t[i+1]==t[-1] and x==t[-1]) else 0.0
   if t[i+p] == t[i]:
      c1 = 0.0
   else:
      c1 = (x - t[i])/(t[i+p] - t[i]) * B(x, p-1, i, t)
   if t[i+p+1] == t[i+1]:
      c2 = 0.0
   else:
      c2 = (t[i+p+1] - x)/(t[i+p+1] - t[i+1]) * B(x, p-1, i+1, t)
   return c1 + c2
# following: define bspline function to produce the fitting curve
# x: parametric values t:knot vector c: coefficient p: degree of the curve

def bspline_x(x, t, c, p):
   n = len(t) - p - 1
   assert (n >= p+1)
   # print('this is lenc',len(c),n)
   assert (len(c) >= n)
   return sum(c[i][0]*B(x, p, i, t) for i in range(len(c)))

def bspline_y(x, t, c, p):
   n = len(t) - p - 1
   assert (n >= p+1)
   assert (len(c) >= n)
   return sum(c[i][1]*B(x, p, i, t) for i in range(len(c)))

def solve_knot_vector(parameter,p,n,start_de,end_de,knotvector):
    # here all the Formal Parameter is as  Least Squares Bspline Curve Approximation pdf said , so it is different with other part
    # t: chord length method parameter p: degree n: control point, and maximum index of different knot  !!!!!
    # start_de or end_de is the k or l specified derivative at ends
    m=len(parameter)-1
    # knotvector=[]
    assistance=[]  # to record the last p+1 knot vector,
    for i in range(p+1):
        knotvector.append(parameter[0])
        assistance.append(parameter[m])
    nc=n-start_de-end_de
    inc=(m+1)/(nc+1)
    low=0
    high=0
    d=-1
    w=[]
    for i in range(nc+1):
        d=d+inc
        high=int(d+0.5)
        sum=0
        for j in range(low,high+1): sum=sum+parameter[j]
        w.append(sum/(high-low+1))
        low=high+1

    iis=1-start_de
    ie=nc-p+end_de
    r=p
    for i in range(iis,ie+1):
        js=max(0,i)
        je=min(nc,i+p-1)
        r=r+1

        sum=0
        for j in range(js,je+1): sum=sum+w[j]
        knotvector.append(sum/(je-js+1))
    knotvector.extend(assistance)
    return

def smooth_and_get_chord_length(input_coarse,output_chordlength):
    # input coarse boundary and get chordlength ratio list as output
    smooth_boundary=[]
    for i in range(len(input_coarse)):
        if i==0 or i==len(input_coarse)-1:
            smooth_boundary.append([input_coarse[i][0],input_coarse[i][1]])
        elif i==1 or i==len(input_coarse)-2:
            smooth_boundary.append([(input_coarse[i][0]+input_coarse[i-1][0]+input_coarse[i+1][0])/3,(input_coarse[i][1]+input_coarse[i-1][1]+input_coarse[i+1][1])/3])
        else:
            temp_list=[0.2*(input_coarse[i-2][0]+input_coarse[i-1][0]+input_coarse[i][0]+input_coarse[i+1][0]+input_coarse[i+2][0]),\
                       0.2*(input_coarse[i-2][1]+input_coarse[i-1][1]+input_coarse[i][1]+input_coarse[i+1][1]+input_coarse[i+2][1])]
            smooth_boundary.append(temp_list)
    # use chord length method to calculate the parametric length ratio
    # not necessary to calculate the distance between first and last point, because it's not closed but open loop
    sum_distance=0
    for i in range(len(smooth_boundary)-1):
        sum_distance+=math.sqrt((smooth_boundary[i+1][0]-smooth_boundary[i][0])**2+(smooth_boundary[i+1][1]-smooth_boundary[i][1])**2)
    temp_sum=0
    if output_chordlength!=[]:
        raise ValueError('the outputlength should be empty')
    output_chordlength.append(0)
    for i in range(1,len(smooth_boundary)):
        temp_sum+=math.sqrt((smooth_boundary[i][0]-smooth_boundary[i-1][0])**2+(smooth_boundary[i][1]-smooth_boundary[i-1][1])**2)
        output_chordlength.append(temp_sum/sum_distance)
    return

