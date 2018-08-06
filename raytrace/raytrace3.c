#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

/* kep parameterei */
#define H               300
#define W               300
#define MAX_REFLECTIONS 10
#define NUM_OBJECTS     5

/* segedmakrok */
#define INFTY    (1e15)
#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))

/* sajat tipusok */
typedef struct{double x,y,z;} vec3;
typedef struct{char type; vec3 color; double shininess, reflectance; void *params;} object;
typedef struct{vec3 center; double radius;} sphere;
typedef struct{vec3 point; vec3 normal;} plane;
typedef struct{vec3 A,B,C;} triangle;
typedef struct{vec3 pos; vec3 color;} light;
typedef struct{vec3 O; vec3 Q;} camera;

/* vektormuveletek */
vec3 sum(vec3 A, vec3 B){
	vec3 C={A.x+B.x, A.y+B.y, A.z+B.z};
	return C;
}
vec3 diff(vec3 A, vec3 B){
	vec3 C={A.x-B.x, A.y-B.y, A.z-B.z};
	return C;
}
vec3 mul(vec3 A, double b){
	vec3 C={A.x*b, A.y*b, A.z*b};
	return C;
}
vec3 power(vec3 A, double b){
	vec3 C={pow(A.x,b), pow(A.y,b), pow(A.z,b)};
	return C;
}
double dot(vec3 A, vec3 B){
	return A.x*B.x+A.y*B.y+A.z*B.z;
}
vec3 cross(vec3 A, vec3 B){
	vec3 C={A.y*B.z-A.z*B.y, A.z*B.x-A.x*B.z, A.x*B.y-A.y*B.x};
	return C;
}
double norm(vec3 A){
	return sqrt(A.x*A.x+A.y*A.y+A.z*A.z);
}
vec3 normalize(vec3 A){
	return mul(A,1/norm(A));
}

/* az atadott buffert elmenti 24 bpp szinmelysegu BMP kepkent */
void saveBMP(const double *img, uint32_t h, uint32_t w, const char *fileName){
	FILE *fp;
	uint8_t padding=(4-(w*3)%4)%4, hdr[54], pixelBuf[3];
	uint32_t fileSize=h*(w*3+padding)+54, bmpSize=h*(w*3+padding);
	int32_t row, col;
	hdr[ 0]=0x42; hdr[ 1]=0x4D;                             //BM
	hdr[ 2]=fileSize&0xFF; hdr[ 3]=(fileSize>>8)&0xFF; hdr[ 4]=(fileSize>>16)&0xFF; hdr[ 5]=(fileSize>>24)&0xFF; //fajl merete
	hdr[ 6]=0x00; hdr[ 7]=0x00; hdr[ 8]=0x00; hdr[ 9]=0x00; //reserved
	hdr[10]=0x36; hdr[11]=0x00; hdr[12]=0x00; hdr[13]=0x00; //bitterkep ofszet
	hdr[14]=0x28; hdr[15]=0x00; hdr[16]=0x00; hdr[17]=0x00; //DIB header merete
	hdr[18]=w&0xFF; hdr[19]=(w>>8)&0xFF; hdr[20]=(w>>16)&0xFF; hdr[21]=(w>>24)&0xFF; //szelesseg
	hdr[22]=h&0xFF; hdr[23]=(h>>8)&0xFF; hdr[24]=(h>>16)&0xFF; hdr[25]=(h>>24)&0xFF; //magassag
	hdr[26]=0x01; hdr[27]=0x00;                             //planar bitmaphoz (sikok szama)
	hdr[28]=0x18; hdr[29]=0x00;                             //BPP
	hdr[30]=0x00; hdr[31]=0x00; hdr[32]=0x00; hdr[33]=0x00; //tomorites
	hdr[34]=bmpSize&0xFF; hdr[35]=(bmpSize>>8)&0xFF; hdr[36]=(bmpSize>>16)&0xFF; hdr[37]=(bmpSize>>24)&0xFF; //bitterkep merete
	hdr[38]=0x00; hdr[39]=0x00; hdr[40]=0x00; hdr[41]=0x00; //vizszintes felbontas
	hdr[42]=0x00; hdr[43]=0x00; hdr[44]=0x00; hdr[45]=0x00; //fuggoleges felbontas
	hdr[46]=0x00; hdr[47]=0x00; hdr[48]=0x00; hdr[49]=0x00; //szinek szama a palettaban
	hdr[50]=0x00; hdr[51]=0x00; hdr[52]=0x00; hdr[53]=0x00; //felhasznalt szinek szama
	fp=fopen(fileName,"w");
	fwrite(hdr,1,54,fp);
	for(row=h-1;row>=0;row--){
		for(col=0;col<w;col++){
			pixelBuf[0]=(uint8_t)(255*img[(row*W+col)*3+2]); //b
			pixelBuf[1]=(uint8_t)(255*img[(row*W+col)*3+1]); //g
			pixelBuf[2]=(uint8_t)(255*img[(row*W+col)*3+0]); //r
			fwrite(&pixelBuf,3,1,fp);
		}
		if(padding>0){
			fwrite(&pixelBuf,padding,1,fp);
		}
	}
	fclose(fp);
}

/* kiszamolja az objektum OD felegyenessel valo metszespontjanak O-tol vett tavolsagat */
double intersect(vec3 D, vec3 O, const object *obj){
	if(obj->type=='s'){ //gomb
		double a, b, c, d, t1, t2;
		vec3 CO=diff(O,((sphere*)(obj->params))->center);
		a=dot(D,D);
		b=2*dot(D,CO);
		c=dot(CO,CO)-(((sphere*)(obj->params))->radius)*(((sphere*)(obj->params))->radius);
		d=b*b-4*a*c;
		if(d<=0) return INFTY;
		t1=(-b-sqrt(d))/(2*a);
		t2=(-b+sqrt(d))/(2*a);
		if(t1>=0){
			if(t2>=0)
				return (t1<t2)?t1:t2;
			else
				return t1;
		}
		else{
			if(t2>=0)
				return t2;
			else
				return INFTY;
		}
	}
	else if(obj->type=='p'){ //sik
		double d=dot(D,((plane*)(obj->params))->normal);
		if(d==0) return INFTY;
		d=dot(diff(((plane*)(obj->params))->point,O),((plane*)(obj->params))->normal)/d;
		if(d>=0) return d;
		return INFTY;
	}
	else if(obj->type=='t'){ //haromszog
		vec3 AB, AC, G, AO, P;
		double f, u, v, t;
		AB=diff(((triangle*)(obj->params))->B,((triangle*)(obj->params))->A);
		AC=diff(((triangle*)(obj->params))->C,((triangle*)(obj->params))->A);
		G=cross(D,AC);
		f=dot(AB,G);
		if((f>-1e-6)&&(f<1e-6)) return INFTY;
		f=1/f;
		AO=diff(O,((triangle*)(obj->params))->A);
		u=f*dot(AO,G);
		if((u<0)||(u>1)) return INFTY;
		P=cross(AO,AB);
		v=f*dot(D,P);
		if((v<0)||(u+v>1)) return INFTY;
		t=f*dot(AC,P);
		if(t>1e-6) return t;
		return INFTY;
	}
	else{
		printf("Error in intersect(): unknown object type.\n\n");
		exit(1);
	}
}

/* visszaadja az objektum M pontbeli normalvektorat */
vec3 getNormalVector(const object *obj, vec3 M){
	if(obj->type=='s'){ //gomb
		return normalize(diff(M,((sphere*)(obj->params))->center));
	}
	else if(obj->type=='p'){ //sik
		return ((plane*)(obj->params))->normal;
	}
	else if(obj->type=='t'){ //haromszog
		return normalize(cross(diff(((triangle*)(obj->params))->B,((triangle*)(obj->params))->A),diff(((triangle*)(obj->params))->C,((triangle*)(obj->params))->A)));
	}
	else{
		printf("Error in normal(): unknown object type.\n\n");
		exit(1);
	}
}

/* foprogram */
int main(){
	uint32_t i, row, col, ray, objIdx;
	double *img, ti, t, intensity;
	bool foundShadow;
	vec3 D, Di, Oi, M, N, ML, MO, color, color_r;
	/* objektumok hozzaadasa */
	sphere   sphere1  ={{ 0.5, -0.1,  1.5}, 0.4};
	sphere   sphere2  ={{-0.5, -0.1,  3.5}, 0.4};
	sphere   sphere3  ={{-0.3, -0.35, 0.6}, 0.15};
	triangle triangle1={{ 0.8, -0.5,  0.3}, {0.5, -0.5, 0.05}, {0.4, -0.1, 0.3}};
	plane    plane1   ={{ 0.0, -0.5,  0.0}, {0.0,1.0,0.0}};
	object objects[NUM_OBJECTS]={
		{'s', {0.8, 0.0, 0.0}, 50, 0.4, &sphere1},
		{'s', {0.0, 0.0, 0.8}, 50, 0.4, &sphere2},
		{'s', {0.5, 0.5, 0.0}, 50, 0.4, &sphere3},
		{'t', {0.9, 0.9, 0.9}, 50, 0.5, &triangle1},
		{'p', {0.0, 1.0, 0.0}, 50, 0.3, &plane1}
	};
	/* fenyek es kamera */
	vec3 colorAmbient={0.05,0.05,0.05};
	light light1={{5,6,-10}, {1,1,1}};
	camera camera1={{0,0.5,-3}, {0,0,1}};
	/* vegigiteralunk a pixeleken */
	img=(double*)calloc(H*W*3,sizeof(double));
	for(row=0;row<H;row++){
		for(col=0;col<W;col++){
			D.x=-1+2.0/(W-1)*col;
			D.y=H/(double)W-2*H/((double)W*(H-1))*row;
			D.z=camera1.Q.z;
			D=diff(D,camera1.O);
			Di=D;
			Oi=camera1.O;
			intensity=1;
			color.x=color.y=color.z=0;
			for(ray=0;ray<MAX_REFLECTIONS;ray++){
				/* megkeressuk a legkozelebbi metszespontot */
				t=INFTY;
				for(i=0;i<NUM_OBJECTS;i++){
					ti=intersect(Di,Oi,&objects[i]);
					if(ti<t){
						t=ti;
						objIdx=i;
					}
				}
				if(t==INFTY) break; //ez a sugar egyetlen targgyal sem talalkozik
				M=sum(Oi,mul(Di,t));
				N=getNormalVector(&objects[objIdx],M);
				ML=normalize(diff(light1.pos,M));
				MO=normalize(diff(Oi,M));
				/* ha arnyekban van, feketen hagyjuk */
				foundShadow=false;
				for(i=0;i<NUM_OBJECTS;i++){
					if(i==objIdx) continue; //onmagat nem arnyekolja
					if(intersect(ML,sum(M,mul(N,0.0001)),&objects[i])!=INFTY){
						foundShadow=true;
						break;
					}
				}
				if(foundShadow) break;
				/* arnyalas */
				color_r=colorAmbient; //ambiens
				color_r=sum(color_r,mul(objects[objIdx].color,MAX(dot(N,ML),0))); //diffuz
				color_r=sum(color_r,power(mul(light1.color,MAX(dot(N,normalize(sum(ML,MO))),0)),objects[objIdx].shininess)); //spekularis
				color=sum(color,mul(color_r,intensity));				
				/* a sugar tovabbi utjanak kiszamitasa */
				Oi=sum(M,mul(N,0.0001));
				Di=normalize(diff(Di,mul(N,2*dot(Di,N))));
				intensity*=objects[objIdx].reflectance;
			}
			/* ezzel a pixellel kesz vagyunk, beirjuk a bufferbe */
			img[(row*W+col)*3+0]=MIN(color.x,1);
			img[(row*W+col)*3+1]=MIN(color.y,1);
			img[(row*W+col)*3+2]=MIN(color.z,1);
		}
	}
	/* rendereles kesz, elmentjuk a kepet */
	saveBMP(img,H,W,"output3.bmp");
	free(img);
	return 0;
}
