import re

# import xlrd
# import xlwt
import math
import numpy
import numpy as np


def Get_both_boundary_nodes_from_inp(fh, both_boundary_nodes):
    ''' get both inner and outer boundaries '''
    if both_boundary_nodes != []: raise ValueError('check non-empty list')
    NodeList = []
    ElementList = []
    AllElementList = [[0]]
    NodeFlag = 0
    ElementFlag = 0
    countNumber = 0
    for line in fh.readlines():
        if line[0:5] == "*Node":
            NodeFlag = 1
            ElementFlag = 0
            continue
        elif line[0:9] == "*Element,":
            NodeFlag = 0
            ElementFlag = 1
            countNumber = 0
            continue
        elif line[0:5] == "*Nset":
            break

        if NodeFlag == 1:
            print('am reading Node')
            countNumber += 1
            # print(countNumber,"-th",line)
            # regular_v1 = re.findall(r"[-+]?([0-9]*\.?[0 - 9]+)", line)
            regular_v1 = re.findall(r"[-+]?[0-9]*\.?[0-9]+", line)
            regular_v2 = []
            for j in regular_v1:
                regular_v2.append(float(j))
            # print(countNumber,'-th',regular_v1,regular_v2)
            NodeList.append(regular_v2)

        if ElementFlag == 1:
            print('am reading Element')
            countNumber += 1
            regular_v1 = re.findall(r"[-+]?[0-9]*\.?[0-9]+", line)
            templist = []
            for j in regular_v1:
                templist.append(float(j))
            # print(countNumber,'-th element',regular_v1,templist)
            AllElementList.append(templist)
            if line[0:2] != "**":
                ElementList.append(templist)
    # here we have read all the element and nodes information

    TrElementList = list(np.transpose(ElementList))
    del (TrElementList[0])  # delete the No. of the Element, this No. is useless
    LIST = []
    for i in range(len(TrElementList)):
        LIST.extend(TrElementList[i])  # store the No. of all node , node is in this form (No.---(x,y,z))

    ReducedLIST = []
    LIST.sort()  # must sort all the list, then we can skip some repeated node by skipping the number of repeating time
    #print('2666666666 line 82',LIST)


    i = 0
    while i < len(LIST):
        hh = LIST.count(LIST[i])
        if hh < 4:
            ReducedLIST.append(LIST[i])
        i += hh
    #print('kkkkkkkkkk', len(LIST), len(ReducedLIST))
    # print(ReducedLIST)
    NodeList.insert(0, [-1234, -1234, -1234,
                        -1234])  # insert one at beginning to make all No. right and its xy coordinates,
    for tem in ReducedLIST:
        both_boundary_nodes.append(
            [NodeList[int(tem)][1], NodeList[int(tem)][2]])  # templist is the inner/outside boundary nodes
    both_boundary_nodes.sort()
    return

def find_outside_boundary(all_nodes, outside):
    """all_nodes include outer and inner boundary, this function is to find outside boundary"""
    if outside != []: raise ValueError('check emtpy')
    temx = all_nodes[-1][0]
    temy = all_nodes[-1][1]

    # temx=all_nodes[0][0]
    # temx=all_nodes[0][1]
    outside.append((temx, temy))
    x_dir = (1, 1, 0, -1, -1, -1, 0, 1)
    y_dir = (0, 1, 1, 1, 0, -1, -1, -1)
    change_dir = (0, 1, 1, 1, 1, 1, 1, 1)
    dir = 7
    for i in range(len(all_nodes)):
        for inc in change_dir:
            dir = (dir + inc) % 8
            if [temx + x_dir[dir], temy + y_dir[dir]] in all_nodes:
                outside.append((temx + x_dir[dir], temy + y_dir[dir]))
                break
            else:
                continue
        temx = temx + x_dir[dir]
        temy = temy + y_dir[dir]
        if dir % 2 == 0:
            dir = (dir + 7) % 8
        else:
            dir = (dir + 6) % 8

        if outside[-1] == outside[0]:
            break
        if i == len(all_nodes) - 1:
            tr_outside = list(np.transpose(outside))
            # fig, ax = plt.subplots()
            # plt.scatter(tr_new_list[0], tr_new_list[1], s=1.2)
            # plt.plot(tr_outside[0], tr_outside[1])
            # #print(outside)
            # plt.show()
            raise ValueError("not useful")
    outside.pop()   # delete the first overlapped point, beginning and the end
    return

def find_x_min(onelist,x_min_Formal_parameter):
    # this is to find the x_min in the list, x_min_Formal_Parameter
    tr_onelist=list(np.transpose(onelist))
    min_list=[]
    for i in range(len(tr_onelist[0])):
        if tr_onelist[0][i]==min(tr_onelist[0]):
            min_list.append(onelist[i])
    min_list.sort(key=lambda x:x[1])
    x_min_Formal_parameter.extend(min_list[0])
    return

def sort_boundary_from_x_min(onelist,x_min_Formal_parameter,Newlist):
    # this is to re-sort the list starting with x_min
    x_min_Formal_parameter=tuple(x_min_Formal_parameter)
    start_index=onelist.index(x_min_Formal_parameter)
    Newlist[:]=[]
    Newlist.extend(onelist[start_index:])
    Newlist.extend(onelist[:start_index])
    Newlist.append(onelist[start_index])  # to make the circle closed loop
    return

def find_straight_line(listnode,threshold_number,emptylist):
    # listnode is the list to be searched, lengthparater is the threshold number,
    #  emptylist will return the begin and end of each straight line
    s1=0
    stem=1
    xFlag=0
    yFlag=0
    while stem<len(listnode)-1:
        #print('this in line 111',s1,stem)
        if stem-s1==1:
            if listnode[s1][0] == listnode[s1+1][0]:
                xFlag=1
                #print('xFlag==1')
                stem+=1
                continue
            elif listnode[s1][1] == listnode[s1+1][1]:
                yFlag=1
                #print('yFlag==1')
                stem += 1
                continue
            else:
                #print('not xFlag yFlag')
                s1+=1
                stem=s1+1
                continue
        else:
            if xFlag==1 and listnode[s1][0] == listnode[stem][0]:
                stem+=1
            elif yFlag==1 and listnode[s1][1] == listnode[stem][1]:
                stem+=1
            elif stem-s1>=threshold_number:
                emptylist.append((s1,stem))
                s1 = stem
                stem = s1 + 1
                xFlag=0
                yFlag=0
            else:
                s1 = stem
                stem = s1 + 1
                xFlag = 0
                yFlag = 0
    if stem-s1>=threshold_number:
        emptylist.append((s1,stem))
        #print(emptylist)
    return


'''smooth the one part of original coarse boundary into smooth Bspline'''
# control points matrix is solved by the pdf 'Least squares Bspline curve approximation with arbitrary end derivative '
# the pdf said this can solve the end derivative and eliminate the deteration as approximation become closer to interpolation
# although the derivative is still not solved, we can use the knot vector method which was powerful


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
        high = math.floor(d + 0.5)
        sum=0
        for j in range(low,high+1): sum=sum+parameter[j]
        # w.append(sum/(high-low+1))
        try:
            w.append(sum / (high - low + 1))
        except:
            print("w", w, "i", i, 'm', m, 'nc', nc, 'high', high, 'parameter', parameter)

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

def Get_Control(input_coarse, chord_length, p, t, n, outputControlPoint):
    # get control points list but the begin and the end need to add forced points
    # p: degree, t: knot vector, n : control point
    B_matrix = []  # Construct the B_matrix
    for k in range(1, len(input_coarse) - 1):
        temp_list = []
        for i in range(1, n):
            # print('Uparameter k n', len(chord_length), k, n, len(input_coarse) - 1)
            temp_list.append(B(chord_length[k], p, i, t))
        B_matrix.append(temp_list)

    # strictly use the method in the NURBS book
    R_k = [0]
    for k in range(1, len(chord_length) - 1):
        temp_list = []
        # print(k, p, len(chord_length) - 1, len(input_coarse))
        temp_list.append(
            input_coarse[k][0] - B(chord_length[k], p, 0, t) * input_coarse[0][0] - B(chord_length[k], p, n, t) *
            input_coarse[-1][0])
        temp_list.append(
            input_coarse[k][1] - B(chord_length[k], p, 0, t) * input_coarse[0][1] - B(chord_length[k], p, n, t) *
            input_coarse[-1][1])
        R_k.append(temp_list)
    R = []
    # print('thisissssssss', type(B(chord_length[1], p, i, t)))
    # print(len(chord_length))
    for i in range(1, n):
        sum_x = 0
        sum_y = 0
        for k in range(1, len(R_k)):
            # print(i, k)
            sum_x += B(chord_length[k], p, i, t) * R_k[k][0]
            sum_y += B(chord_length[k], p, i, t) * R_k[k][1]
        R.append([sum_x, sum_y])

    B_T = numpy.transpose(B_matrix)
    M = numpy.matmul(B_T, B_matrix)
    C = numpy.linalg.solve(M, R)
    # if outputControlPoint != []:
    #     raise ValueError("this list should be empty")
    C=numpy.array(C).tolist()
    outputControlPoint.extend(C)  # returned value of bspline(x,t,c,p) are array, C need to be matrix, thus numpy.array().tolist()

    return

def okkkkkkkkkkkkkbtain_xy_of_bspline(input_coarse,output_xy_boundary,Control_list,n,degree=3):
    # to adapt teh method with C1-continuity
    U_parameter = []
    smooth_and_get_chord_length(input_coarse, U_parameter)
    p = degree  # p: degree of the curve
    # n=int(len(input_coarse)/8)

    t = []
    solve_knot_vector(U_parameter, p, n, 0, 0, t)

    C_list = []
    Get_Control(input_coarse, U_parameter, p, t, n, C_list)

    C_list.insert(0, input_coarse[0])  # first control point is fixed
    C_list.append(input_coarse[-1])  # last control point is fixed
    # print(len(C_list), 'this is length of list', n)
    Control_list.extend(C_list)
    xx = numpy.linspace(0.000, 1, 1000)
    if output_xy_boundary!=[]:
        raise ValueError('it should be empty')
    for x in xx:
        output_xy_boundary.append((bspline_x(x, t, C_list, p), bspline_y(x, t, C_list, p)))  # this is a list of tuples

    ts_coordinate_bspline=list(numpy.transpose(output_xy_boundary))
    ts_C=list(numpy.transpose(Control_list))
    # fig, ax = plt.subplots()
    # ax.plot(ts_C[0], ts_C[1], 'o')  # --------------------------------
    # ax.plot(ts_C[0], ts_C[1], '--', lw=0.5, label='control point and line')
    # ax.plot(ts_coordinate_bspline[0], ts_coordinate_bspline[1], 'r-', lw=0.25, label='Bspline')
    # plt.title('overall outside boundary')
    # ax.grid(True)
    # plt.show()

    return

def merge_straght_line(s_to_e,new_s_to_e):
    # to avoid missing the last part we should check the  ~~~'' s_to_e[-1][1] is len(outside)-1 ''

    if new_s_to_e != []: raise ValueError('check this')
    s_to_e1=[]
    s_to_e2 = []
    for i in range(len(s_to_e)):
        s_to_e1.append(s_to_e[i][0])
        s_to_e1.append(s_to_e[i][1])

    for tem in s_to_e1:
        if s_to_e1.count(tem)>1:
            continue
        else:s_to_e2.append(tem)

    if len(s_to_e2)%2!=0:
        #print(s_to_e2)
        raise ValueError('this must be wrong')
    for i in range(int(len(s_to_e2)/2)):
        new_s_to_e.append((s_to_e2[2*i],s_to_e2[2*i+1]))
    # if s_to_e[-1][1]!=len(outside)-1:
    #     new_s_to_e.append((s_to_e2[2 * i], s_to_e2[2 * i + 1]))
    # print(new_s_to_e)
    return

def skkkkkkkmooth_boundary_by_connecting(outside, s_to_e,new_outside):
    '''ATTENTION: this is a total new one'''
    # connecting smooth part with part of original boundary
    if new_outside!=[]: raise ValueError('check this')
    #print(outside)
    for ii in range(len(s_to_e)-1):
        new_outside.extend(outside[s_to_e[ii][0]:s_to_e[ii][1]])

        coarse=outside[s_to_e[ii][1]-1:s_to_e[ii+1][0]+1]   # here  s_to_e[ii+1][0]+1  the "+1 " is added (5-2 date)
        output=[]
        control=[]
        n=int((s_to_e[ii+1][0]-s_to_e[ii][1])/10)+3
        obtain_xy_of_bspline(coarse,output,control,n)
        #print('22222222 new_outside',new_outside[-4],new_outside[-3],new_outside[-2],new_outside[-1],'next',output[0],output[1],'next',output[-2],output[-1],'next',outside[s_to_e[ii+1][0]-1],outside[s_to_e[ii+1][0]])
        new_outside.extend(output)
    if s_to_e[-1][1]!=len(outside)-1:
        new_outside.extend(outside[s_to_e[-1][0]:s_to_e[-1][1]])   #  !!!!!! attention this is a new added line
        coarse = outside[s_to_e[ - 1][1]:]
        #print('777777',coarse)
        output = []
        control = []
        n = int((s_to_e[ii][0] - s_to_e[ii - 1][1]) / 10) + 3
        obtain_xy_of_bspline(coarse, output, control, n)
        new_outside.extend(output)
    else:
        new_outside.extend(outside[s_to_e[ - 1][1]:])
    return
# ----------------following is the new one
def obtain_xy_of_bspline(input_coarse, output_xy_boundary, Control_list, n, former, latter, degree=int(3), c1=False):
    # to adapt the method with C1-continuity
    U_parameter = []
    smooth_and_get_chord_length(input_coarse, U_parameter)
    p = degree  # p: degree of the curve
    # n=int(len(input_coarse)/8)

    t = []
    solve_knot_vector(U_parameter, p, n, 0, 0, t)

    C_list = []
    Get_Control(input_coarse, U_parameter, p, t, n, C_list)

    C_list.insert(0, input_coarse[0])  # first control point is fixed
    C_list.append(input_coarse[-1])  # last control point is fixed
    # print(len(C_list), 'this is length of list', n)
    # following is to make the C-1 continuity by project the second control point into the line
    # the former or letter value is set before calling this function
    measure = math.sqrt((C_list[0][0] - C_list[-1][0]) ** 2 + (C_list[0][1] - C_list[-1][1]) ** 2) / len(C_list)
    measure /= 2

    # Updated on 23/05/19 by dedao.
    # Update the algorithm for moving the endpoint for c1 continuity:
    c1 = True
    if c1 == True:
        C_list[1][0] = C_list[0][0] + measure * (C_list[0][0] - former[0]) / math.sqrt(
            (C_list[0][0] - former[0]) ** 2 + (C_list[0][1] - former[1]) ** 2)
        C_list[1][1] = C_list[0][1] + measure * (C_list[0][1] - former[1]) / math.sqrt(
            (C_list[0][0] - former[0]) ** 2 + (C_list[0][1] - former[1]) ** 2)

        C_list[-2][0] = C_list[-1][0] + measure * (C_list[-1][0] - latter[0]) / math.sqrt(
            (C_list[-1][0] - latter[0]) ** 2 + (C_list[-1][1] - former[1]) ** 2)
        C_list[-2][1] = C_list[-1][1] + measure * (C_list[-1][1] - latter[1]) / math.sqrt(
            (C_list[-1][0] - latter[0]) ** 2 + (C_list[-1][1] - former[1]) ** 2)

    Control_list.extend(C_list)
    xx = numpy.linspace(0.000, 1, 100)
    if output_xy_boundary!=[]:
        raise ValueError('it should be empty')
    for x in xx:
        output_xy_boundary.append((bspline_x(x, t, C_list, p), bspline_y(x, t, C_list, p)))  # this is a list of tuples

    ts_coordinate_bspline=list(numpy.transpose(output_xy_boundary))
    ts_C=list(numpy.transpose(Control_list))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(ts_C[0], ts_C[1], 'o')  # --------------------------------
    ax.plot(ts_C[0], ts_C[1], '--', lw=0.5, label='control point and line')
    ax.plot(ts_coordinate_bspline[0], ts_coordinate_bspline[1], 'r-', lw=0.25, label='Bspline')
    ax.plot(former[0], former[1], 'ro')
    ax.plot(latter[0], latter[1], 'go')
    plt.title('part of outside boundary')
    ax.grid(True)
    plt.show()

    return

def smooth_boundary_by_connecting(outside, s_to_e,new_outside):
    '''ATTENTION: this is a total new one'''
    # connecting smooth part with part of original boundary
    if new_outside!=[]: raise ValueError('check this')
    #print(outside)
    if s_to_e[0][0] == 0:
        pass
    else:
        coarse=outside[0:s_to_e[0][0]+1]
        output=[]
        control=[]
        n=int((s_to_e[0][0]+1)/10)+3
        former=outside[-1]
        latter=outside[s_to_e[0][0]+1]  # in this 'for ' loop, latter must be valid
        #print('qqqqqqq',n,coarse)
        obtain_xy_of_bspline(coarse,output,control,n,former,latter)
        #print('22222222 new_outside',new_outside[-4],new_outside[-3],new_outside[-2],new_outside[-1],'next',output[0],output[1],'next',output[-2],output[-1],'next',outside[s_to_e[ii+1][0]-1],outside[s_to_e[ii+1][0]])
        new_outside.extend(output)
    for ii in range(len(s_to_e)-1):
        new_outside.extend(outside[s_to_e[ii][0]:s_to_e[ii][1]])
        coarse=outside[s_to_e[ii][1]-1:s_to_e[ii+1][0]+1]   # here  s_to_e[ii+1][0]+1  the "+1 " is added (5-2 date)
        output=[]
        control=[]
        n=int((s_to_e[ii+1][0]-s_to_e[ii][1])/10)+3
        if s_to_e[ii][1]-1 >0:  # s_to_e[ii][1] is the coarse begin
            former=outside[s_to_e[ii][1]-2]
        elif outside[0][0]==outside[1][0] or outside[0][1]==outside[1][1]:  # 按照一开始的前两个点优化
            former=outside[1]
        else: former=(2*outside[1][0]-outside[0][0],2*outside[1][1]-outside[0][1])
        latter=outside[s_to_e[ii+1][0]+1]  # in this 'for ' loop, latter must be valid
        #print('qqqqqqq',n,coarse)
        obtain_xy_of_bspline(coarse,output,control,n,former,latter)
        #print('22222222 new_outside',new_outside[-4],new_outside[-3],new_outside[-2],new_outside[-1],'next',output[0],output[1],'next',output[-2],output[-1],'next',outside[s_to_e[ii+1][0]-1],outside[s_to_e[ii+1][0]])
        new_outside.extend(output)
        print("the node list length", len(outside))
    if s_to_e[-1][1]!=len(outside)-1:
        print("the last spline piece")
        new_outside.extend(outside[s_to_e[-1][0]:s_to_e[-1][1]])   #  !!!!!! attention this is a new added line
        coarse = outside[s_to_e[-1][1]-1:]
        #print('777777',coarse)
        output = []
        control = []
        n = int((s_to_e[ii][0] - s_to_e[ii - 1][1]) / 10) + 3
        former=outside[s_to_e[-1][1]-2]  # in this if section, former must be valid
        latter=(2*outside[-2][0]-outside[-1][0],2*outside[-2][1]-outside[-1][1])
        obtain_xy_of_bspline(coarse, output, control, n, former, latter)
        new_outside.extend(output)
    else:
        print("the last straight line")
        new_outside.extend(outside[s_to_e[-1][0]:])
    return


def get_final_boundary(outside, original_outside_nodes_dict, c1):
    # (outside, s_to_e,new_outside):
    '''The final boundary consists of many points
    input:
    output: list of coords of points [(x,y,z)]
    '''
    # connecting smooth part with part of original boundary
    #boundary_piece_type = None
    # outside = [(b, c) for a, b, c in outside_boundary_coords]
    if original_outside_nodes_dict == None:
        s_to_e = [(0, len(outside)-1)]
    else:
        s_to_e = []
        # print(sorted(original_outside_nodes_dict.keys()))

        check_index = 0
        check_in_list = True
        while check_in_list:
            if check_index in original_outside_nodes_dict.keys():
                start_of_piece = check_index
                check_in_piece = True
                while check_in_piece:
                    check_index += 1
                    print(check_index)
                    if not check_index in original_outside_nodes_dict.keys():
                        check_in_piece = False
                        end_of_piece = check_index - 1
                        s_to_e.append((start_of_piece, end_of_piece))
                    if check_index >= sorted(original_outside_nodes_dict.keys())[-1]:
                        check_in_list = False
            else:
                check_in_piece = False
                while not check_in_piece:
                    check_index += 1
                    print(check_index)
                    if check_index in original_outside_nodes_dict.keys():
                        check_in_piece = True
                    if check_index >= sorted(original_outside_nodes_dict.keys())[-1]:
                        check_in_list = False
        #checking_index = sorted(original_outside_nodes_dict.keys())[0]
        #for node_index in sorted(original_outside_nodes_dict.keys()):
        #    if node_index == checking_index:
        #        first_checking_index = checking_index
        #        checking_index += 1
        #    else:
        #        s_to_e.append((first_checking_index, checking_index-1))
    print("check list")
    print(s_to_e)
    new_outside = []

    degree = 3
    # original code_pack
    # if new_outside!=[]: raise ValueError('check this')
    #print(outside)
    if s_to_e[0][0] == 0:
        pass
    else:
        coarse=outside[0:s_to_e[0][0]+1]
        output=[]
        control=[]
        n=int((s_to_e[0][0]+1)/10)+3
        former=outside[-1]
        latter = outside[s_to_e[0][0] + 2]  # in this 'for ' loop, latter must be valid

        obtain_xy_of_bspline(coarse, output, control, n, former, latter, degree, c1)
        #print('22222222 new_outside',new_outside[-4],new_outside[-3],new_outside[-2],new_outside[-1],'next',output[0],output[1],'next',output[-2],output[-1],'next',outside[s_to_e[ii+1][0]-1],outside[s_to_e[ii+1][0]])
        new_outside.extend(output)
    for ii in range(len(s_to_e)-1):
        # adding the nodes in the middle
        new_outside.extend(outside[s_to_e[ii][0]:s_to_e[ii][1]])
        # adding keep nodes
        coarse=outside[s_to_e[ii][1]-1:s_to_e[ii+1][0]+1]   # here  s_to_e[ii+1][0]+1  the "+1 " is added (5-2 date)
        output=[]
        control=[]
        n=int((s_to_e[ii+1][0]-s_to_e[ii][1])/10)+3
        # if s_to_e[ii][1]-1 >0:
        former = outside[s_to_e[ii][1] - 2]
        # elif outside[0][0]==outside[1][0] or outside[0][1]==outside[1][1]:
        #     former=outside[1]
        # else: former=(2*outside[1][0]-outside[0][0],2*outside[1][1]-outside[0][1])
        latter = outside[s_to_e[ii + 1][0] + 2]  # in this 'for ' loop, latter must be valid
        obtain_xy_of_bspline(coarse, output, control, n, former, latter, degree, c1)
        new_outside.extend(output)
        print("the node list length", len(outside))
    if s_to_e[-1][1]!=len(outside)-1:
        print("the last spline piece")
        new_outside.extend(outside[s_to_e[-1][0]:s_to_e[-1][1]])
        coarse = outside[s_to_e[-1][1]-1:]
        output = []
        control = []
        n = int((s_to_e[ii][0] - s_to_e[ii - 1][1]) / 10) + 3
        former = outside[s_to_e[-1][1] - 2]
        latter = outside[1]
        obtain_xy_of_bspline(coarse, output, control, n, former, latter, degree, c1)
        new_outside.extend(output)
    else:
        print("the last straight line")
        new_outside.extend(outside[s_to_e[-1][0]:])
    return new_outside
