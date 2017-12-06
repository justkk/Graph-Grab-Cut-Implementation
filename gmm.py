import cv2;
import numpy as np;
import math;
import maxflow;
import copy;
import sys;
import os;

class gmp(object):

	def __init__(self,img,fgdata,bgdata):
		self.img=img;
		self.fgdata=fgdata;
		self.bgdata=bgdata;
		self.potentials={};
		self.lables={};
		a=self.img.shape;
		self.rows=a[0];
		self.cols=a[1];

	def train_fg(self):
		samples={};
		for i in self.fgdata:
			samples[i]=self.img[i[0],i[1]];
		key=samples.keys();
		data=[];
		for i in key:
			data.append((samples[i]));
		data=np.asarray(data);
		em=cv2.EM();
		retval,log,lables,probs=em.train(data,None,None,None);
		#print probs;
		return em,log;

	def train_bg(self):
		samples={};
		for i in self.bgdata:
			samples[i]=self.img[i[0],i[1]];
		key=samples.keys();
		data=[];
		for i in key:
			data.append(samples[i]);
		data=np.asarray(data);
		em=cv2.EM();
		retval,log,lables,probs=em.train(data,None,None,None);
		#print probs;
		return em,log;
	def findvalue(self,fgem,bgem,flog,blog):
		# 0->background;
		# 1->foreground;
		a=self.img.shape;
		rows=a[0];
		cols=a[1];
		for i in range(0,rows):
			for j in range(0,cols):
				fsample,fprobs=fgem.predict(img[i,j]);
				bsample,bprobs=bgem.predict(img[i,j]);
				key=(i,j);
				fprobs=fprobs[0];
				bprobs=bprobs[0];
				fprobs=[1-h for h in fprobs];
				bprobs=[1-h for h in bprobs];
				fproduct=1;
				bproduct=1;
				for h in fprobs:
					fproduct*=h;

				for h in bprobs:
					bproduct*=h;
				#print fsample,bsample;

				#self.potentials[key]=[-1*math.log(0.0000000000001+fproduct),-1*math.log(0.0000000000001+bproduct)];
				self.potentials[key]=[-1*fsample[0],-1*bsample[0]];


		return 0;
	def rer(self,node):
		return node[0]*self.cols+node[1];
	
	def addpairwise(self,node1,node2):
		tt=0;
		node1=self.img[node1[0],node1[1]];
		node2=self.img[node2[0],node2[1]];
		



		for i in range(1,len(node1)):
			tt+=pow(float(node1[i])-float(node2[i]),2);
		tt=math.sqrt(tt);
		value=pow(2.314,-1*(tt/2))
		return 2+3*value;
			
			

	def buildgraph(self):
		g = maxflow.Graph[float](self.rows*self.cols+2, 10000);
		a=self.img.shape;
		rows=a[0];
		cols=a[1];
		num_nodes=rows*cols;
		nodes=g.add_nodes(num_nodes);
		for i in range(0,rows):
			for j in range(0,cols):
				
				present=[i,j];
				up=[i,j-1];
				down=[i,j+1];
				left=[i-1,j];
				right=[i+1,j];

				g.add_tedge(self.rer(present),self.potentials[(i,j)][0],self.potentials[(i,j)][1]);
				
				if(j-1>=0):
					cost=self.addpairwise(present,up)
					g.add_edge(nodes[self.rer(present)],nodes[self.rer(up)],cost,cost);

				if(j+1<cols):
					cost=self.addpairwise(present,down)
					g.add_edge(nodes[self.rer(present)],nodes[self.rer(down)],cost,cost);

				if(i-1>=0):
					cost=self.addpairwise(present,left)
					g.add_edge(nodes[self.rer(present)],nodes[self.rer(left)],cost,cost);

				if(i+1<rows):
					cost=self.addpairwise(present,right)
					g.add_edge(nodes[self.rer(present)],nodes[self.rer(right)],cost,cost);

		print g.maxflow();	
		newimage=copy.copy(self.img);
		count=0;
		for i in range(0,rows):
			for j in range(0,cols):
				key=(i,j);
				self.lables[key]=g.get_segment(nodes[self.rer([i,j])]);
				if(self.lables[key]==1):
					newimage[i,j]=img1[i,j];
					count+=1;
				else:
					newimage[i,j]=[0,0,0];
		cv2.imshow('newimage',newimage);
		cv2.waitKey(0);
		cv2.destroyAllWindows();
		print count







print sys.argv[1];
image=cv2.imread(str(sys.argv[1]));
img=cv2.imread(str(sys.argv[1]));
img1=cv2.imread(str(sys.argv[1]));

def run(s1,e1,s2,e2):
	global img;
	img=cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
	a=[];
	b=[];
	r=img.shape;
	c=r[1];
	r=r[0];
	for i in range(0,r):
		for j in range(0,c):
			if (i in range(s1,e1)) and (j in range(s2,e2)):
				a.append(tuple([i,j]));
			else:
				b.append(tuple([i,j]));
	
	gm=gmp(img,a,b);
	for k in range(0,5):
		fgem,flog=gm.train_fg();
		bgem,blog=gm.train_bg();
		gm.findvalue(fgem,bgem,flog,blog)
		gm.buildgraph();
		a=[];
		b=[];
		for i in range(0,r):
			for j in range(0,c):
				if(gm.lables[(i,j)]==1):
					a.append(tuple([i,j]));
				else:
					b.append(tuple([i,j]));
		gm.fgdata=a;
		gm.bgdata=b;
		print len(a),len(b);
























	
drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
lx,ly = -1,-1
k=0
def draw_circle(event,x,y,flags,param):
	global ix,iy,lx,ly,drawing,mode
	if event == cv2.EVENT_LBUTTONDOWN :
		drawing = True
		ix,iy = x,y
	elif event == cv2.EVENT_LBUTTONUP:
		drawing = False
		if mode == True:
			cv2.rectangle(image,(ix,iy),(x,y),(0,0,0),2)
			lx=x;
			ly=y;
			k=27;
			cv2.destroyAllWindows()
			#cv2.setMouseCallback('',None)
			run(iy,ly,ix,lx);
			sys.exit(0);
      		

cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
	cv2.imshow('image',image)
	k = cv2.waitKey(1) & 0xFF
	if k == ord('m'):
		mode = mode
	elif k == 27:
		break

cv2.destroyAllWindows()


