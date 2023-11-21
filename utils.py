# -*- coding: utf-8 -*-
from PyQt5.QtCore import QVariant, Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QMessageBox, QProgressBar
from qgis.core import *
import os.path
import numpy as np
import triangle as tr
import matplotlib.pyplot as plt
import math
from scipy.spatial import ConvexHull
import scipy.spatial
import networkx as nx
from sklearn import preprocessing

def data_organize(features):
    edge_id=0
    segmentList=[]
    pointList=[]
    edgeIDList = []
    startEndPointEdge = dict()
    for feature in features:
        # 边设置成feature的num要素每个feature一个num
        edge_id=feature.attribute("num")
        # edge_id = feature.attribute("num2")
        try:
            pointXYList = feature.geometry().asMultiPolyline()[0]
            firstPoint = pointXYList[0]
            lastPoint = pointXYList[-1]
        except:
            pointXYList = feature.geometry().asMultiPolygon()[0][0]
            firstPoint = pointXYList[0]
            lastPoint = pointXYList[-1]
        # if firstPoint not in startEndPointEdge.keys():
        #     startEndPointEdge.update({firstPoint:[edge_id]})
        # else:
        #     startEndPointEdge[firstPoint].append(edge_id)
        #
        # if lastPoint not in startEndPointEdge.keys():
        #     startEndPointEdge.update({lastPoint: [edge_id]})
        # else:
        #
        #     startEndPointEdge[lastPoint].append(edge_id)
        for i in range(len(pointXYList)):
            addPointFTag = True
            addSegment = True
            point = pointXYList[i]
            if i==0:
                addSegment = False
                if firstPoint in pointList:
                    lastPointIndex = pointList.index(firstPoint)
                    addPointFTag = False
                    continue
                else:
                    lastPointIndex = len(pointList)
            elif i==1 and i!=len(pointXYList) - 1:
                currentPointIndex = len(pointList)
            elif i == len(pointXYList) - 1:
                if lastPoint in pointList:
                    addPointFTag = False
                    currentPointIndex = pointList.index(lastPoint)
                    if len(pointXYList) == 2:
                        lastPointIndex = pointList.index(firstPoint)
                    else:
                        lastPointIndex = len(pointList)-1
                else:
                    currentPointIndex = len(pointList)
                    if len(pointXYList) == 2:
                        lastPointIndex = pointList.index(firstPoint)
                    else:
                        lastPointIndex = currentPointIndex-1
            else:
                currentPointIndex = len(pointList)
                lastPointIndex = currentPointIndex - 1
            if addPointFTag:
                pointList.append(point)
                edgeIDList.append(edge_id)

            if addSegment:
                segment = [lastPointIndex,currentPointIndex]
                segmentList.append(segment)

    vertices = np.array(pointList)
    segments = np.array(segmentList)
    buildDict = dict({'vertices':vertices,'segments':segments})
    deletKeys=[]

    for key,value in startEndPointEdge.items():
        if len(value)==1:
            deletKeys.append(key)
    for key in deletKeys:
        startEndPointEdge.pop(key)
    return segmentList,pointList,edgeIDList,buildDict,startEndPointEdge

def filt_tris(trainagules,pointList,polintList2,edgeIDList,startEndPointEdge):
    tris = trainagules['triangles'].tolist()
    deleteTri = []
    revisesTris = trainagules['triangles'].tolist()
    # 筛选建筑拓扑错误
    xy = trainagules['vertices'].tolist()
    XY = []
    llist = []
    for i in pointList:
        XY.append([i[0], i[1]])
    for i in xy:
        if i not in XY:
            llist.append(i)
    for tri in tris:
        # for i in tri:
        #     print(trainagules["vertices"][i])
        points = [pointList[i] for i in tri]
        edges = [edgeIDList[i] for i in tri]
        if all(element in polintList2 for element in points):
            revisesTris.remove(tri)
            deleteTri.append(tri)
            continue
        if len(set(edges))==1:
            revisesTris.remove(tri)
            deleteTri.append(tri)   #如果三点在同一个三角形中则去掉，表明是内部点
        elif len(set(edges))==2:
            for i in range(3):
                if points[i] in startEndPointEdge.keys() and len(startEndPointEdge[points[i]])>1:
                    _edges = copy.deepcopy(edges)
                    _edges.remove(edges[i])
                    _edges =_edges+startEndPointEdge[points[i]]
                    if len(set(_edges)) == len (startEndPointEdge[points[i]]):
                        count = [_edges.count(i) for i in _edges]
                        if 3 in count:
                            revisesTris.remove(tri)
                            deleteTri.append(tri)
                            continue

    trainagules.update({'triangles':np.array(revisesTris)})
    polygonList = []
    for tri in revisesTris:
        polygonList.append(QgsGeometry.fromPolygonXY([[pointList[int(i)] for i in tri]]))
    # for tri in tris:
    #     polygonList.append(QgsGeometry.fromPolygonXY([[pointList[int(i)] for i in tri]]))
    return revisesTris,len(revisesTris),polygonList

def mininumAreaRectangle(A):
    size, min_area = len(A), 1.7976931348623157e+128
    i, j = 0, size - 1
    d, u0, u1 = [0, 0], [0, 0], [0, 0]
    OBB = OBBOject()
    while i < size:
        length_edge = math.sqrt(pow(A[i][0] - A[j][0], 2) +
                                pow(A[i][1] - A[j][1], 2))
        if length_edge != 0:
            u0[0], u0[1] = (A[i][0] - A[j][0]) / length_edge, (A[i][1] - A[j][1]) / length_edge
            u1[0], u1[1] = 0 - u0[1], u0[0]
            # u0和u1是垂直的且是单位向量,e0,e1是长和宽的一半
            # print(math.sqrt(u0[0]*u0[0]+u0[1]*u0[1]),u0[0]*u1[0]+u0[1]*u1[1])
            min0, max0, min1, max1, minX_i, maxX_i, minY_i, maxY_i = 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0
            for k in range(0, size):
                d[0], d[1] = A[k][0] - A[j][0], A[k][1] - A[j][1]
                # The projection onto the u0,u1.
                dotU0, dotU1 = Dot(d, u0), Dot(d, u1)
                if dotU0 < min0:
                    min0, minX_i = dotU0, k
                if dotU0 > max0:
                    max0, maxX_i = dotU0, k
                if dotU1 < min1:
                    min1, minY_i = dotU1, k
                if dotU1 > max1:
                    max1, maxY_i = dotU1, k

            area = (max0 - min0) * (max1 - min1)
            if area < min_area:
                min_area = area
                # Update the information.
                OBB.c[0] = A[j][0] + (u0[0] * (max0 + min0) + u1[0] * (max1 + min1)) * 0.5
                OBB.c[1] = A[j][1] + (u0[1] * (max0 + min0) + u1[1] * (max1 + min1)) * 0.5
                OBB.u0, OBB.u1 = [u0[0], u0[1]], [u1[0], u1[1]]
                OBB.e0, OBB.e1 = (max0 - min0) * 0.5, (max1 - min1) * 0.5
                OBB.minX_i, OBB.maxX_i, OBB.minY_i, OBB.maxY_i = minX_i, maxX_i, minY_i, maxY_i
        j, i = i, i + 1
    return OBB, min_area

def get_mean_radius(A,CX,CY):
    mean_radius=0
    for i in range(0,len(A)-1):
        mean_radius+=math.sqrt(math.pow(A[i][0]-CX,2)+math.pow(A[i][1]-CY,2))
    N=len(A)-1
    return mean_radius/N

def dett(point1,point2,point3):
    return (point2[0]-point1[0])*(point3[1]-point1[1])-(point2[1]-point1[1])*(point3[0]-point1[0])

def get_basic_parametries_of_Poly(A):
	CX,CY,area,peri=0,0,0,0
	# if(len(A)<1):
	# 	raise Exception('ILLEGAL_ARGUMENT')
	# 	return [[CX,CY],area,peri]
	# closure the polygon.
	# if(A[0][0]!=A[len(A)-1][0] or A[0][1]!=A[len(A)-1][1]):
	# 	A.append(A[0])
	# calculate the center point [CX,CY] and perometry L.
	for i in range(0,len(A)-1):
		CX+=A[i][0]
		CY+=A[i][1]
		peri+=math.sqrt(pow(A[i+1][0]-A[i][0],2)+pow(A[i+1][1]-A[i][1],2))
	CX=CX/(len(A)-1)
	CY=CY/(len(A)-1)
	#calculate the area.
	# if(len(A)<3):
	# 	raise Exception('ILLEGAL_ARGUMENT')
	# 	return [[CX,CY],area,peri]
	indication_point=A[0]
	for i in range(1,len(A)-1):
		area+=dett(indication_point,A[i],A[i+1])
	return [[CX,CY],abs(area)*0.5,abs(peri)]

#多边形面积
def polygon_area(points):
    n = len(points)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1] - points[j][0] * points[i][1]
    return abs(area) / 2

class OBBOject:
    def __init__(self):
        self.u0, self.u1, self.c = [0, 0], [0, 0], [0, 0]
        self.e0, self.e1 = 0.0, 0.0
        # *_i represents index.
        self.minX_i, self.maxX_i = 0.0, 0.0
        self.minY_i, self.maxY_i = 0.0, 0.0

    # convert the descriptors to rectangle points.
    def toVertexes(self, isClose=True):
        vertexex = []
        vertexex.append([self.c[0] + self.u0[0] * self.e0 + self.u1[0] * self.e1,
                         self.c[1] + self.u0[1] * self.e0 + self.u1[1] * self.e1])
        vertexex.append([self.c[0] + self.u0[0] * self.e0 - self.u1[0] * self.e1,
                         self.c[1] + self.u0[1] * self.e0 - self.u1[1] * self.e1])
        vertexex.append([self.c[0] - self.u0[0] * self.e0 - self.u1[0] * self.e1,
                         self.c[1] - self.u0[1] * self.e0 - self.u1[1] * self.e1])
        vertexex.append([self.c[0] - self.u0[0] * self.e0 + self.u1[0] * self.e1,
                         self.c[1] - self.u0[1] * self.e0 + self.u1[1] * self.e1])
        if isClose:
            vertexex.append([self.c[0] + self.u0[0] * self.e0 + self.u1[0] * self.e1,
                             self.c[1] + self.u0[1] * self.e0 + self.u1[1] * self.e1])
        return vertexex

    # calculate the point (index) that touched rectangle with the maximum X.
    # should pass the cooridates as an argument.
    def pointTouchRectWithMaxX(self, A):
        max_X, max_P = A[self.minX_i][0], self.minX_i
        if A[self.maxX_i][0] > max_X:
            max_X, max_P = A[self.maxX_i][0], self.maxX_i
        if A[self.minY_i][0] > max_X:
            max_X, max_P = A[self.minY_i][0], self.minY_i
        if A[self.maxY_i][0] > max_X:
            max_X, max_P = A[self.maxY_i][0], self.maxY_i
        return max_P

    # def distanceOfPointFromRect(self, P):
    #     vertexex = self.toVertexes()
    #     min_Dis = pointToLine(P, vertexex[3], vertexex[0])
    #     for i in range(0, 3):
    #         if pointToLine(P, vertexex[i], vertexex[i + 1]) < min_Dis:
    #             min_Dis = pointToLine(P, vertexex[i], vertexex[i + 1])
    #     return min_Dis

    def Orientation(self):
        if self.e0 > self.e1:
            if self.u0[1] > 0:
                return math.acos(self.u0[0])
            else:
                return math.pi - math.acos(self.u0[0])
        else:
            if self.u1[1] > 0:
                return math.acos(self.u1[0])
            else:
                return math.pi - math.acos(self.u1[0])

class Point(object):
    def __init__(self,x,y):
        self.x=x
        self.y=y
    def distance(self,otherpoint):
        dx=self.x-otherpoint.x
        dy=self.y-otherpoint.y
        return math.sqrt(dx**2+dy**2)
    def triangle_area(self, p2, p3):
        x1, y1 = self.x,self.y
        x2, y2 = p2.x,p2.y
        x3, y3 = p3.x,p3.y
        return 0.5 * abs(x1 * y2 + x2 * y3 + x3 * y1 - y1 * x2 - y2 * x3 - y3 * x1)

def Dot(u, v):
    return u[0] * v[0] + u[1] * v[1]

def proj_length_A(a, b):
    proj = np.dot(a, b) / np.dot(b, b) * b
    proj_length_A = np.linalg.norm(proj)
    return proj_length_A   #输出a在b上的投影长度

def geometric_feature(bulding_feature,convexhull_path):  #计算建筑物的几何特征
    feature_num = len(bulding_feature)
    Node_coordinates,building_geometric_feature,building_boundary=[],[],[]
    for i in range(feature_num):
        feature_geometry = bulding_feature[i].geometry().asMultiPolygon()[0][0]
        building_coordinates = []
        for j in range(0, len(feature_geometry)):
            building_coordinates.append(
                [feature_geometry[j][0], feature_geometry[j][1]])  # building_coordinates是各个建筑物的坐标集
        building_boundary.append(building_coordinates)
        [[CX, CY], area, peri] = get_basic_parametries_of_Poly(building_coordinates)
        uniform_coords = np.array([[(j[0] - CX), (j[1] - CY)] for j in building_coordinates])
        # feature:area,peri,circularity,mean_radius,orientation,concavity,MBRFullness,Elongation,WALL_orien
        circularity = 4 * math.pi * area / math.pow(peri, 2)
        mean_radius = get_mean_radius(building_coordinates, CX, CY)
        OBB, SMBR_area= mininumAreaRectangle(building_coordinates)
        orientation = OBB.Orientation()
        convexHull = ConvexHull(uniform_coords)
        ConvexHull_peri = convexHull.area
        ConvexHull_area = convexHull.volume
        #area： 输入维度 > 2 时凸包的表面积。当输入points 为二维时，这是凸包的周长。
        # volume： 浮点数输入维度 > 2 时凸包的体积。当输入points 为二维时，这是凸包的面积。
        long_edge = 0
        distance, long_edge_from, long_edge_to = scipy.spatial.distance.cdist(uniform_coords[convexHull.vertices],
                                                                              uniform_coords[convexHull.vertices],
                                                                              'euclidean'), 0, 0
        for i in range(0, len(distance)):
            for j in range(i, len(distance)):
                if distance[i, j] > long_edge: long_edge, long_edge_from, long_edge_to = distance[i, j], i, j
        concavity_area = area / ConvexHull_area
        concavity_peri = peri / ConvexHull_peri
        MBRFullness = area / SMBR_area
        Elongation = OBB.e1 / OBB.e0 if OBB.e0 > OBB.e1 else OBB.e0 / OBB.e1
        edge_orien_weight, edge_length_sum = 0, 0
        for j in range(len(uniform_coords) - 1):
            dx, dy = uniform_coords[j + 1][0] - uniform_coords[j][0], uniform_coords[j + 1][1] - uniform_coords[j][
                1]
            edge_orien = (math.atan2(dy, dx) + 2 * math.pi) % math.pi
            edge_length = math.sqrt(dx * dx + dy * dy)
            edge_orien_weight += edge_orien * edge_length
            edge_length_sum += edge_length
        WALL_orien = edge_orien_weight / edge_length_sum
        Node_coordinates.append([CX,CY])
        feature= [area, peri, circularity, mean_radius, orientation, concavity_area,concavity_peri, MBRFullness, Elongation,
                         WALL_orien]
        building_geometric_feature.append(feature)
    return  building_geometric_feature,Node_coordinates,building_boundary

def calCN_Junction(node,centerNode,startEndPointEdge,calCenterNodeJunction):
    if node in startEndPointEdge.keys():
        if node not in calCenterNodeJunction.keys():
            calCenterNodeJunction.update({node: [centerNode]})
        else:
            if centerNode not in calCenterNodeJunction[node]:
                calCenterNodeJunction[node].append(centerNode)

#创建中点
def center_point(point1, point2):
    return QgsPointXY((point1.x() + point2.x()) / 2, (point1.y() + point2.y()) / 2)

def creat_skeleton(revisesTris,nerbors,pointList,startEndPointEdge,trisNum,edgeIDList,boundaryIS=None):
    polylineList = []
    calCenterNodeJunction = dict()
    mean_dis = {}
    vis_dis = {}
    convex_dis = {}
    point_index = {}
    for index, value in enumerate(edgeIDList):
        point_index.setdefault(value, []).append([pointList[index][0],pointList[index][1]])
    for i in range(trisNum):
        tri = revisesTris[i]
        triNerbor = nerbors[i]
        triNodeCoord = [pointList[j] for j in tri]
        nerborTrisToNode = np.array(np.where(triNerbor > -1))
        nerborsNum = nerborTrisToNode.shape[1]
        # region 似乎是为了处理又boundary存在的情况下，处理边界三角形的，只适用于建筑物,并且建筑物一定要完全位于边界内
        if boundaryIS!=None:
            maxId= max(edgeIDList)
            edgeIDIndex = [pointList.index(i) for i in triNodeCoord]
            edgeID = [edgeIDList[i] for i in edgeIDIndex]
            if maxId in edgeID and nerborsNum==3 and len(set(edgeID))==3:
                ID = [0, 1, 2]
                nodeFaceToSegmentID = edgeID.index(maxId)
                ID.remove(nodeFaceToSegmentID)
                fistNerborSegmentNodeID = ID[0]
                secondNerborSegmentNodeID = ID[1]
                nodeFaceToSegment = triNodeCoord[nodeFaceToSegmentID]
                fistNerborSegmentNode = triNodeCoord[fistNerborSegmentNodeID]
                secondNerborSegmentNode = triNodeCoord[secondNerborSegmentNodeID]
                centerNode = center_point(fistNerborSegmentNode, secondNerborSegmentNode)
                polylineList.append(QgsGeometry.fromPolylineXY([nodeFaceToSegment, centerNode]))
                if nodeFaceToSegment in startEndPointEdge.keys():
                    try:
                        startEndPointEdge[nodeFaceToSegment].remove(startEndPointEdge[nodeFaceToSegment][0])
                    except:
                        print('{0}报错,三角形为{1}'.format(nodeFaceToSegment,i))
                continue
            elif maxId in edgeID:
                continue
        # endregion
        if nerborsNum == 1:
            ID = [0, 1, 2]
            nodeFaceToSegmentID = nerborTrisToNode[0, 0]
            ID.remove(nodeFaceToSegmentID)
            fistNerborSegmentNodeID = ID[0]
            secondNerborSegmentNodeID = ID[1]
            nodeFaceToSegment = triNodeCoord[nodeFaceToSegmentID]
            fistNerborSegmentNode = triNodeCoord[fistNerborSegmentNodeID]
            secondNerborSegmentNode = triNodeCoord[secondNerborSegmentNodeID]
            centerNode = center_point(fistNerborSegmentNode, secondNerborSegmentNode)
            polylineList.append(QgsGeometry.fromPolylineXY([nodeFaceToSegment, centerNode]))
            if nodeFaceToSegment in startEndPointEdge.keys():
                try:
                    startEndPointEdge[nodeFaceToSegment].remove(startEndPointEdge[nodeFaceToSegment][0])
                except:
                    print('{0}报错,三角形为{1}'.format(nodeFaceToSegment,i))
        elif nerborsNum == 2:
            ID = [0, 1, 2]
            firstNodeFaceToSegmentID = nerborTrisToNode[0, 0]
            secondNodeFaceToSegmentID = nerborTrisToNode[0, 1]
            ID.remove(nerborTrisToNode[0, 0])
            ID.remove(nerborTrisToNode[0, 1])
            thirdNodeID = ID[0]
            firstNodeFaceToSegment = triNodeCoord[firstNodeFaceToSegmentID]
            secondNodeFaceToSegment = triNodeCoord[secondNodeFaceToSegmentID]
            thirdNode = triNodeCoord[thirdNodeID]
            P1 = Point(firstNodeFaceToSegment[0], firstNodeFaceToSegment[1])
            P2 = Point(secondNodeFaceToSegment[0], secondNodeFaceToSegment[1])
            P3 = Point(thirdNode[0], thirdNode[1])
            tris = [edgeIDList[m] for m in revisesTris[i]]
            # 添加边关系
            edgelink = []
            for elem in tris:
                if elem not in edgelink:
                    edgelink.append(elem)
            edgelink2 = [edgelink[1], edgelink[0]]
            single_mean_distance = (Point.distance(P1, P3) + Point.distance(P2, P3)) / 2.0
            single_tri_area = Point.triangle_area(P1, P2, P3)
            centerNode1 = center_point(firstNodeFaceToSegment, thirdNode)
            centerNode2 = center_point(secondNodeFaceToSegment, thirdNode)
            center_node1 = Point(centerNode1[0], centerNode1[1])
            center_node2 = Point(centerNode2[0], centerNode2[1])
            single_vis_distance = Point.distance(center_node1, center_node2)    #vis_distance是中点距离
            # print(edgelink)
            if tuple(edgelink) in mean_dis or tuple(edgelink2) in mean_dis:
                mean_dis[tuple(edgelink)].append(single_mean_distance)
                mean_dis[tuple(edgelink2)].append(single_mean_distance)
            else:
                mean_dis[tuple(edgelink)] = [single_mean_distance]
                mean_dis[tuple(edgelink2)] = [single_mean_distance]
            if tuple(edgelink) in vis_dis or tuple(edgelink2) in vis_dis:
                vis_dis[tuple(edgelink)].append(single_vis_distance)
                vis_dis[tuple(edgelink2)].append(single_vis_distance)
            else:
                vis_dis[tuple(edgelink)] = [single_vis_distance]
                vis_dis[tuple(edgelink2)] = [single_vis_distance]
            if tuple(edgelink) in convex_dis or tuple(edgelink2) in convex_dis:
                convex_dis[tuple(edgelink)].append(single_tri_area)
                convex_dis[tuple(edgelink2)].append(single_tri_area)
            else:
                convex_dis[tuple(edgelink)] = [single_tri_area]
                convex_dis[tuple(edgelink2)] = [single_tri_area]
            polylineList.append(QgsGeometry.fromPolylineXY([centerNode1, centerNode2]))
            if thirdNode in startEndPointEdge.keys():
                if thirdNode not in calCenterNodeJunction.keys():
                    calCenterNodeJunction.update({thirdNode: [centerNode1, centerNode2]})
                else:
                    if centerNode1 not in calCenterNodeJunction[thirdNode]:
                        calCenterNodeJunction[thirdNode].append(centerNode1)
                    if centerNode2 not in calCenterNodeJunction[thirdNode]:
                        calCenterNodeJunction[thirdNode].append(centerNode2)
            # if firstNodeFaceToSegment in startEndPointEdge.keys():
            #     if firstNodeFaceToSegment not in calCenterNodeJunction.keys():
            #         calCenterNodeJunction.update({firstNodeFaceToSegment: [centerNode1]})
            #     else:
            #         if centerNode1 not in calCenterNodeJunction[firstNodeFaceToSegment]:
            #             calCenterNodeJunction[firstNodeFaceToSegment].append(centerNode1)
            # if secondNodeFaceToSegment in startEndPointEdge.keys():
            #     if secondNodeFaceToSegment not in calCenterNodeJunction.keys():
            #         calCenterNodeJunction.update({secondNodeFaceToSegment: [centerNode2]})
            #     else:
            #         if centerNode2 not in calCenterNodeJunction[secondNodeFaceToSegment]:
            #             calCenterNodeJunction[secondNodeFaceToSegment].append(centerNode2)
            calCN_Junction(firstNodeFaceToSegment, centerNode1, startEndPointEdge, calCenterNodeJunction)
            calCN_Junction(secondNodeFaceToSegment, centerNode2, startEndPointEdge, calCenterNodeJunction)
        elif nerborsNum == 3:
            firstNode = triNodeCoord[0]
            secondNode = triNodeCoord[1]
            thirdNode = triNodeCoord[2]
            barycenter = barycenter_point(firstNode, secondNode, thirdNode)
            centerNode1 = center_point(firstNode, secondNode)
            centerNode2 = center_point(firstNode, thirdNode)
            centerNode3 = center_point(thirdNode, secondNode)
            polylineList.append(QgsGeometry.fromPolylineXY([barycenter, centerNode1]))
            polylineList.append(QgsGeometry.fromPolylineXY([barycenter, centerNode2]))
            polylineList.append(QgsGeometry.fromPolylineXY([barycenter, centerNode3]))
            calCN_Junction(firstNode, centerNode1, startEndPointEdge, calCenterNodeJunction)
            calCN_Junction(secondNode, centerNode2, startEndPointEdge, calCenterNodeJunction)
            calCN_Junction(thirdNode, centerNode3, startEndPointEdge, calCenterNodeJunction)
    for key in mean_dis:
        values = mean_dis[key]  # 获取当前键的值
        avg_value = sum(values) / len(values)  # 计算平均值
        mean_dis[key] = avg_value  # 将当前键的值替换为平均值,平均距离
    for key in vis_dis:
        values2 = vis_dis[key]  # 获取当前键的值
        avg_value2 = sum(values2)  # 计算平均值
        vis_dis[key] = avg_value2  # 将当前键的值替换为平均值,视觉距离
    for key in convex_dis:
        values3 = convex_dis[key]  # 获取当前键的值
        avg_value3 = sum(values3)  # 计算平均值
        feature1_index,feature2_index=key[0],key[1]
        feature1_coord,feature2_coord=point_index[feature1_index],point_index[feature2_index]
        feature1_area,feature2_area=polygon_area(feature1_coord),polygon_area(feature2_coord)
        convex_coord=feature1_coord+feature2_coord
        convex=ConvexHull(convex_coord)
        new_value=(feature1_area+feature2_area+avg_value3)/convex.volume
        convex_dis[key] = new_value # 将当前键的值替换为平均值,凸包距离
    weight_dis = {}
    for key in vis_dis.keys() | mean_dis.keys():
        weight_dis[key] = vis_dis.get(key, 0) / mean_dis.get(key, 0)
    # 需要平均距离mean_dis,视觉距离vis_dis,凸包距离convex_dis,加权距离weight_dis
    return calCenterNodeJunction,polylineList,mean_dis,vis_dis,convex_dis,weight_dis

def satics_tris(revisesTris,segmentList,trisNum,pointList):
    # <editor-fold desc="统计点属于哪些">
    nodeToTris = dict()
    triNode = np.array(revisesTris).reshape(-1)
    pointtopoint=[]
    for i in range(len(pointList)):
        triIndex = np.array(np.where(triNode == i)).reshape(-1) // 3
        nodeToTris.update({i: triIndex.tolist()})
        # node对应的三角形索引
    # </editor-fold>

    # <editor-fold desc="统计邻近三角形关系">
    tris= np.array(revisesTris)
    nerbors = np.ones([trisNum,3])*-1
    segmentSet = [set(i) for i in segmentList]
    for i in range(trisNum):
        for j in range(3):
            otherTwoNode = tris[i].tolist()
            otherTwoNode.remove(tris[i,j])
            pointtopoint.append(otherTwoNode)
            # exteriorEdge = True
            trisIndex = nodeToTris[otherTwoNode[0]] + nodeToTris[otherTwoNode[1]]
            for k in trisIndex:
                if set(otherTwoNode)<set(revisesTris[k]):
                    if k!=i and set(otherTwoNode) not in segmentSet:
                        nerbors[i,j]=k

            #             exteriorEdge = False
            # if exteriorEdge and set(otherTwoNode) in cSegmentSet:
            #     nerbors[i, j] = trisNum
    # </editor-fold>
    return nerbors,nodeToTris,pointtopoint

#创建重心
def barycenter_point(point1, point2, point3):
    return QgsPointXY((point1.x() + point2.x() + point3.x()) / 3, (point1.y() + point2.y() + point3.y()) / 3)

def revise_skeleton(segmentList,startEndPointEdge,calCenterNodeJunction,pointList,nodeToTris,revisesTris,nerbors,polylineList):
    segmentArray = np.array(segmentList).reshape(-1)
    _segmentList = np.array(segmentList)
    for key, value in startEndPointEdge.items():
        if len(value) == 0:
            continue
        else:
            processNum = len(value)
            try:
                keyCenNodeList = calCenterNodeJunction[key]
            except:
                print(key)
                continue
            distantListToKey = [QgsGeometry.fromPolylineXY([key, i]).length() for i in keyCenNodeList]
        if processNum == 1:
            minIndex = distantListToKey.index(min(distantListToKey))
            polylineList.append(QgsGeometry.fromPolylineXY([key, keyCenNodeList[minIndex]]))
        else:
            keyNodeID = pointList.index(key)
            connectSegments = [_segmentList[i // 2, :].tolist() for i in
                               np.where(segmentArray == keyNodeID)[0].tolist()]
            connectSegmentsArrar = copy.deepcopy(connectSegments)
            keyNodeIDTris = nodeToTris[keyNodeID]
            connectSegmentsTris = []
            # 查询以该汇流点为顶点的三角形
            connectSegmentsTris = copy.deepcopy(keyNodeIDTris)
            # for connectSegment in connectSegments:
            #     connectSegment.remove(keyNodeID)
            #     otherIDTris = nodeToTris[connectSegment[0]]
            #     connectSegmentsTris = connectSegmentsTris + list(set(keyNodeIDTris).intersection(set(otherIDTris)))
            # connectSegmentsTris = list(set(connectSegmentsTris))
            tri1 = []
            for i in connectSegmentsTris:
                if len(np.where(nerbors[i] > -1)[0]) == 1:
                    if nerbors[i][revisesTris[i].index(keyNodeID)] > -1:
                        tri1.append(revisesTris[i])
            tri1EdgeCount = [0 for i in range(len(connectSegmentsArrar))]
            tri1Edge = []
            for i in range(len(connectSegmentsArrar)):
                cs = connectSegmentsArrar[i]
                for tri in tri1:
                    if set(cs) < set(tri):
                        tri1EdgeCount[i] += 1
                        tri1Edge.append(cs)

            # <editor-fold desc="老版本的寻找targetEdge">
            # targetEdge = []
            #筛选不是1三角形的边
            # for edge in tri1Edge:
            #     if edge not in connectSegmentsArrar:
            #         targetEdge.append(edge)
            # </editor-fold>

            targetEdgeIndex = [i for i in range(len(tri1EdgeCount)) if tri1EdgeCount[i]==0]
            targetEdge = [connectSegmentsArrar[i] for i in targetEdgeIndex]
            # 存在targetEdge和不存在的情况，targetEdge是未参与构建1三角形的边
            if len(targetEdge) == 0:
                targetEdge = [connectSegmentsArrar[i] for i in range(len(connectSegmentsArrar)-1)]
            i = 0
            while i < processNum:
                try:
                    minIndex = distantListToKey.index(min(distantListToKey))
                except:
                    print("minIndex is wrong:{0}".format(key))
                    break
                centernode = keyCenNodeList[minIndex]
                polylineList.append(QgsGeometry.fromPolylineXY([key, centernode]))
                tag=[]
                for et in targetEdge:
                    tag.append(judge_node_segment_direction(centernode, [pointList[et[0]], pointList[et[1]]]))
                # tag = judge_node_segment_direction(centernode, [pointList[targetEdge[0]], pointList[targetEdge[1]]])
                keyCenNodeList.remove(centernode)
                distantListToKey.remove(distantListToKey[minIndex])
                deletNodes = []
                for node in keyCenNodeList:
                    _tag = []
                    for et in targetEdge:
                        _tag.append(judge_node_segment_direction(node, [pointList[et[0]], pointList[et[1]]]))
                    # _tag = judge_node_segment_direction(node, [pointList[targetEdge[0]], pointList[targetEdge[1]]])
                    if tag == _tag:
                        deletNodes.append(node)
                for node in deletNodes:
                    distantListToKey.remove(distantListToKey[keyCenNodeList.index(node)])
                    keyCenNodeList.remove(node)
                i += 1
    return polylineList

def write_shapeFile(input_layer,shapeFileName,polygonList,geometryType = QgsWkbTypes.Polygon):

    fields = QgsFields()
    # crs = QgsProject.instance().crs()
    crs = input_layer.crs()
    transform_context = QgsProject.instance().transformContext()
    save_options = QgsVectorFileWriter.SaveVectorOptions()
    save_options.driverName = "ESRI Shapefile"
    save_options.fileEncoding = "UTF-8"

    writer = QgsVectorFileWriter.create(
        shapeFileName,
        fields,
        geometryType,
        crs,
        transform_context,
        save_options
    )
    if writer.hasError() != QgsVectorFileWriter.NoError:
        print("Error when creating shapefile: ", writer.errorMessage())
    for polygon in polygonList:
        fet = QgsFeature()
        fet.setGeometry(polygon)
        # fet.setAttributes([1, "text"])
        writer.addFeature(fet)
    del writer

def shift_degree(OBB,point):
    shifting_degree_node = OBB.c
    shifting_degree_vector = [shifting_degree_node[0] - point[-1][0], shifting_degree_node[1] - point[-1][1]]
    SBR_vector_length = np.array([math.cos(OBB.Orientation()), math.sin(OBB.Orientation())])
    rotation_matrix = np.array([[0, -1], [1, 0]])
    SBR_vector_width=np.dot(rotation_matrix,SBR_vector_length)
    shifting_degree_vector = np.array(shifting_degree_vector)
    S1 = proj_length_A(shifting_degree_vector, SBR_vector_length)
    S2 = proj_length_A(shifting_degree_vector, SBR_vector_width)
    if OBB.e0>OBB.e1:
        return 2*S1/(OBB.e0+1e-5),2*S2/(OBB.e1+1e-5)
    else:
        return 2*S1/(OBB.e1+1e-5),2*S2/(OBB.e0+1e-5)

def run(folder):
    # <editor-fold desc="数据读取">
    print('open the folder:{}'.format(folder))
    print("输入建筑物数据和范围,统计几何特征")
    folder_list=os.listdir(folder)
    folder_list.remove("预处理阶段")
    folder_list.remove("模型后处理")
    folder_list.remove("实验阶段")
    folder_list.remove("实验阶段2.0")
    folder_list.remove("实验阶段3.0")
    for i in range(len(folder_list)):
      if i==0:
        id=folder_list[i]
        print('open the folder:{}'.format(id))
        folder=os.path.join(folder,id)
        building_path=os.path.join(folder,"building")+".shp"
        partition_path=os.path.join(folder,"partition")+".shp"
        convexhull_path=os.path.join(folder,"convexhull")+".shp"
        tri_path=os.path.join(folder,"tri")+".shp"
        skeleton_path=os.path.join(folder,"skeleton")+".shp"
        building_layer = QgsVectorLayer(building_path, 'building', 'ogr')
        partition_layer = QgsVectorLayer(partition_path, 'partition', 'ogr')
        bulding_feature= list(building_layer.dataProvider().getFeatures())
        partition_feature= list(partition_layer.dataProvider().getFeatures())
        building_geometric_feature,Node_coordinates,building_boundary= geometric_feature(bulding_feature, convexhull_path) #建筑物的十三个几何特征
        features=bulding_feature+partition_feature
        point_List = []
        for feature in features:
            pointXYList = feature.geometry().asMultiPolygon()[0][0]
            # pointXYList=pointXYList[:-1]
            point_List.append(pointXYList)
        # </editor-fold>
        print('数据组织')
        segmentList, pointList, edgeIDList, buildDict, startEndPointEdge = data_organize(features)
        segmentList2, pointList2, edgeIDList2, buildDict2, startEndPointEdge2 = data_organize(partition_feature)
        print("构建三角网图")
        # <editor-fold desc="构建图">
        trainagules = tr.triangulate(buildDict,"pc") #pc

        # print("拓扑检查")
        # # 筛选建筑拓扑错误
        # xy = buildDict['vertices'].tolist()
        # XY = []
        # llist=[]
        # for i in pointList:
        #     XY.append([i[0], i[1]])
        # for i in xy:
        #     if i not in XY:
        #         llist.append(i)
        # print(llist)

        # tr.plot(plt.axes(), **trainagules,dpi=800)
        # plt.show()
        # </editor-fold>
        print('剔除自组织三角网')
        revisesTris, trisNum, polygonList = filt_tris(trainagules, pointList,pointList2, edgeIDList, startEndPointEdge)
        print('统计三角形邻近关系')
        # 统计三角形邻近关系
        nerbors, nodeToTris, pointtopoint = satics_tris(revisesTris, segmentList, trisNum, pointList)
        features_num=int(len(features))
        adjacency_matrix=np.zeros((features_num,features_num))
        for i in range(len(pointtopoint)):
            firstpoint, secondpoint=edgeIDList[pointtopoint[i][0]],edgeIDList[pointtopoint[i][1]]
            adjacency_matrix[firstpoint, secondpoint] = 1
            adjacency_matrix[secondpoint, firstpoint] = 1
        partition_num=[-i for i in range(1,len(partition_feature)+1)]
        adjacency_matrix=np.delete(adjacency_matrix,partition_num,axis=0)
        adjacency_matrix = np.delete(adjacency_matrix, partition_num, axis=1)   #领接矩阵去除边界的邻接关系
        print("生成连接边")
        edge = []
        for i in range(len(adjacency_matrix)):
            for j in range(i+1,len(adjacency_matrix)):
                if adjacency_matrix[i,j]==1:
                    edge.append([i,j])
                    edge.append([j,i])
        building_graph=nx.Graph()
        building_graph.add_edges_from(edge)
        nx.draw(building_graph,with_labels=True)
        plt.show()
        for i in range(len(building_graph)):
            print(i)
            building_graph.nodes[i]["coordinate"]=Node_coordinates[i]
        building_graph_feature=[]
        for i in range(len(building_graph)):
            first_order_neighbor=[n for n in building_graph.neighbors(i)]
            degree=len(first_order_neighbor)
            first_order_neighbor.append(i)
            first_order_neighbor_coordinate=[building_graph.nodes[n]["coordinate"] for n in first_order_neighbor]
            OBB, SMBR_area= mininumAreaRectangle(first_order_neighbor_coordinate)
            shifting_degree_l,shifting_degree_w=shift_degree(OBB, first_order_neighbor_coordinate)
            building_graph_feature.append([degree,shifting_degree_l,shifting_degree_w])
        print("生成图并计算长宽偏移度特征")
        print('初步构建骨架')
        # 初步构建骨架
        calCenterNodeJunction, polylineList,mean_dis,vis_dis,convex_dis,weight_dis = creat_skeleton(revisesTris, nerbors, pointList, startEndPointEdge,
                                                             trisNum, edgeIDList, partition_path)
        print('修正汇流点处骨架')
        # 进一步修正汇流交点处的估计
        polylineList = revise_skeleton(segmentList, startEndPointEdge, calCenterNodeJunction, pointList, nodeToTris,
                                       revisesTris, nerbors, polylineList)
        for i in range(len(partition_feature)):
            partition_line=partition_feature[i].geometry().asMultiPolygon()[0][0]
            polylineList.append(QgsGeometry.fromPolylineXY(partition_line))
        print('添加Block边界,文件写出')
        write_shapeFile(building_layer, shapeFileName=tri_path, polygonList=polygonList, geometryType=QgsWkbTypes.Polygon)
        write_shapeFile(building_layer, shapeFileName=skeleton_path, polygonList=polylineList,
                        geometryType=QgsWkbTypes.LineString)
        building_features=[]
        for i in range(len(bulding_feature)):
            building_feature=[]
            for j in range(len(building_geometric_feature[i])):
                building_feature.append(building_geometric_feature[i][j])
            for k in range(len(building_graph_feature[i])):
                building_feature.append(building_graph_feature[i][k])
            building_features.append(building_feature)
        area, peri, circularity, mean_radius, orientation, concavity_area, concavity_peri, MBRFullness, Elongation,WALL_orien,degree,shifting_degree_l,shifting_degree_w=[],[],[],[],[],[],[],[],[],[],[],[],[]
        for i in range(len(building_features)):
            area.append(building_features[i][0])
            peri.append(building_features[i][1])
            circularity.append(building_features[i][2])
            mean_radius.append(building_features[i][3])
            orientation.append(building_features[i][4])
            concavity_area.append(building_features[i][5])
            concavity_peri.append(building_features[i][6])
            MBRFullness.append(building_features[i][7])
            Elongation.append(building_features[i][8])
            WALL_orien.append(building_features[i][9])
            degree.append(building_features[i][10])
            shifting_degree_l.append(building_features[i][11])
            shifting_degree_w.append(building_features[i][12])
        scaler = preprocessing.StandardScaler()
        area = scaler.fit_transform(np.array(area).reshape(-1, 1))
        peri = scaler.fit_transform(np.array(peri).reshape(-1, 1))
        circularity = scaler.fit_transform(np.array(circularity).reshape(-1, 1))
        mean_radius = scaler.fit_transform(np.array(mean_radius).reshape(-1, 1))
        orientation = scaler.fit_transform(np.array(orientation).reshape(-1, 1))
        concavity_area = scaler.fit_transform(np.array(concavity_area).reshape(-1, 1))
        concavity_peri = scaler.fit_transform(np.array(concavity_peri).reshape(-1, 1))
        MBRFullness = scaler.fit_transform(np.array(MBRFullness).reshape(-1, 1))
        Elongation = scaler.fit_transform(np.array(Elongation).reshape(-1, 1))
        WALL_orien = (np.array(WALL_orien).reshape(-1, 1))
        degree = scaler.fit_transform(np.array(degree).reshape(-1, 1))
        shifting_degree_l = scaler.fit_transform(np.array(shifting_degree_l).reshape(-1, 1))
        shifting_degree_w = scaler.fit_transform(np.array(shifting_degree_w).reshape(-1, 1))
        x = np.array(
            [area, peri, circularity, mean_radius, orientation, concavity_area,concavity_peri, MBRFullness, Elongation, WALL_orien,degree,shifting_degree_l,shifting_degree_w]).T
        x = x.reshape(x.shape[1], x.shape[2])
        np.save("C:/Users/28634/Desktop/GCN论文/实验部分/实验阶段4.0/feature/building.npy",x)
        print("归一化并存储建筑物特征至npy")
        print("计算边特征:最短距离")
        edge_distance = {}
        for i in range(len(edge)):
            from_point,to_point=edge[i][0],edge[i][1]
            distance,min_edge=scipy.spatial.distance.cdist(building_boundary[from_point],
                                                                              building_boundary[to_point],
                                                                              'euclidean'),1.7976931348623157e+128
            for i in range(0, len(distance)):
                for j in range(0, len(distance[i])):
                    if distance[i, j] < min_edge: min_edge = distance[i, j]
            key=[from_point,to_point]
            key=tuple(key)
            edge_distance[key]=min_edge
        # vis_dis,convex_dis,weight_dis
        edge_feature = {}
        for key in edge_distance.keys():
            if key in vis_dis.keys() and key in convex_dis.keys() and key in weight_dis.keys() and key in mean_dis.keys():
                edge_feature[key]=[edge_distance[key],mean_dis[key],vis_dis[key],convex_dis[key],weight_dis[key]]
        edge_feature=[[*key,*value] for key,value in edge_feature.items()]
        # for key in vis_dis.keys() | convex_dis.keys()| weight_dis.keys()| edge_distance.keys() |mean_dis.keys():
        #     edge_feature[key] = [edge_distance.get(key,0),mean_dis.get(key, 0),vis_dis.get(key, 0),convex_dis.get(key, 0),weight_dis.get(key,0)]
        edge_txt = open("C:/Users/28634/Desktop/GCN论文/实验部分/实验阶段4.0/feature/edge.txt", "w")
        for i in edge_feature:
            edge_txt.write(str(i) + '\n')
        edge_txt.close()
        print("存储边矩阵至txt")





if __name__=="__main__":
    folder=r"C:\Users\28634\Desktop\GCN论文\实验部分"
    run(folder)
    # 注意要建立feature和result