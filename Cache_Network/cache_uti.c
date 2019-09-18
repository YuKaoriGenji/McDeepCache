#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#define W 20
#define H 20
#define BOUNDX (240/W)
#define BOUNDY (320/H)
#define NUMX 0
#define NUMY 1 
#define WIDTH 2
#define HEIGHT 3
#define MX 4
#define MY 5
#define SKIP 1
#define RATE 0.6

float result[57*77*512]={0};
typedef struct StructPointerTest
{
    int numx;
    int numy;
    int width;
    int height;
    int Mx;
    int My;
}StructPointerTest,*StructPointer;

float* testFunc(float* pict1,float* pict2,float* pict3,float* pict4,float* center,int left,int top, int right, int down){
    float * respo=result;
    long count=0;
    for(long i=0;i<57*top*512;i++){
        result[i]=pict1[i];
        count++;
    }
    printf("count: %ld\n",count);
    for(int i=top;i<=down;i++){
        for(int j=0;j<left*512;j++){
            result[i*57*512+j]=pict2[(i-top)*left*512+j];
            count++;
        }
        for(int j=0;j<(right+1-left)*512;j++){
            //printf("i:%d,j:%d total:%d\n",i,j,i*(right-left+1)*512+j);
            result[i*57*512+j+left*512]=center[(i-top)*(right-left+1)*512+j];
            count++;
        }
        for(int j=0;j<(57-right-1)*512;j++){
            result[i*57*512+j+(right+1)*512]=pict3[(i-top)*(57-right-1)*512];
            count++;
        }
    }
    printf("next count: %ld\n",count);
    /*
    for(long i=0;i<57*(77-down-1)*512;i++){
        //result[(down+1)*57*512+i]=pict4[i];
        printf("%ld\n",(down+1)*57*512+i);
        result[(down+1)*57*512+i]=0;
    }
    */

    return result;
}

double PSNR(double* pict1,double* pict2,int* block1,int* block2){
    double diff=0;
    double average=0;
    double rmse=0;
    //printf("I am here 1!\n");
    if(block2[NUMX]<0 || block2[NUMX]>=BOUNDX ||block2[NUMY]<0 || block2[NUMY]>=BOUNDY){
        return 0;
    }
    for(int i=0;i<block1[WIDTH];i++){
        for(int j=0;j<block1[HEIGHT];j++){
            for(int k=0;k<3;k++){
                diff+=(pict1[(i+block1[WIDTH]*block1[NUMX])*320*3+(j+block1[HEIGHT]*block1[NUMY])*3+k]-pict2[(i+block2[WIDTH]*block2[NUMX])*320*3+(j+block2[HEIGHT]*block2[NUMY])*3+k])*(pict1[(i+block1[WIDTH]*block1[NUMX])*320*3+(j+block1[HEIGHT]*block1[NUMY])*3+k]-pict2[(i+block2[WIDTH]*block2[NUMX])*320*3+(j+block2[HEIGHT]*block2[NUMY])*3+k]);
            }
        }
    }
    average=diff/((block1[WIDTH]*block1[HEIGHT])*3);
    //printf("average:%f\n",average);
    rmse=sqrt(average);
    //printf("1/rmse:%f\n",20*log10(1.0/rmse));
    //printf("rmse:%f\n",(rmse));
    if(rmse!=0){
        //printf("I am here! 3\n");
        return (20*log10(1.0/rmse));
    }
    else{
        return 10000;
    }
}

int* Diamond_Search(double* pict1,int* block1,double* previous_image){
    int x=block1[NUMX];
    int y=block1[NUMY];
    float Big_max=0;
    int Big_num=0;
    int x_s=0;
    int y_s=0;
    int Small_num=0;
    float Small_max=0;
    static int result[2]={0};
    int target[6]={0};
    int Big_scale[9][2]={{x,y},{x-1,y-1},{x+1,y-1},{x,y-2},{x-2,y},{x-1,y+1},{x,y+2},{x+2,y},{x-1,y-1}};
    double Big_score[9]={0};
    for(int i=0;i<9;i++){
        //target[6]={Big_scale[i][0],Big_scale[i][1],W,H,0,0};
        target[0]=Big_scale[i][0];
        target[1]=Big_scale[i][1];
        //printf("target[0]:%d,target[1]:%d\n",target[0],target[1]);
        target[2]=W;
        target[3]=H;
        target[4]=0;
        target[5]=0;
        Big_score[i]=PSNR(pict1,previous_image,block1,target);
        //printf("Big_score[%d]:%f\n",i,Big_score[i]);
        if(Big_score[i]>Big_max){
            Big_max=Big_score[i];
            Big_num=i;
        }
    }
    x_s=Big_scale[Small_num][0];
    y_s=Big_scale[Small_num][1];
    int Small_scale[5][2]={{x_s,y_s},{x_s-1,y_s},{x_s+1,y_s},{x_s,y_s-1},{x_s,y_s+1}};
    double Small_score[5]={0};
    for(int i=0;i<5;i++){

        //target[6]={Small_scale[i][0],Small_scale[i][1],W,H,0,0};
        target[0]=Small_scale[i][0];
        target[1]=Small_scale[i][1];
        target[2]=W;
        target[3]=H;
        target[4]=0;
        target[5]=0;
        Small_score[i]=PSNR(pict1,previous_image,block1,target);
        //printf("target[0]:%d,target[1]:%d\n",target[0],target[1]);
        if(Small_score[i]>Small_max){
            Small_max=Small_score[i];
            Small_num=i;
        }
    }
    result[0]=Small_scale[Small_num][0];
    result[1]=Small_scale[Small_num][1];
    return result;
    //return p;

}

double density(int(* List)[6],int left,int top, int right, int down){
    int cardinal=(right-left+1)*(down-top+1);
    int effective=0;
    if(cardinal==0){
        return 0;
    }
    //printf("List[3][WIDTH]:%d\n",List[3][WIDTH]);
    for(int i=0;(i<200 && List[i][WIDTH]!=0);i++){
        if(List[i][NUMX]>=left && List[i][NUMX]<right && List[i][NUMY]>=top && List[i][NUMY]<down){
            //printf("One item over\n");
            effective++;
        }
    }
    return ((double)effective/(double)cardinal);
}

StructPointer ComputeOverlay(double* picture1,double* picture2){
    int tao=14;
    int total_num=0;
    int sum_x=0;
    int sum_y=0;
    int result[200][6]={0};
    int pict1[6]={0,0,W,H,0,0};
    int pict2[6]={0,0,W,H,0,0};
    int pict_one[6]={0,0,W,H,0,0};
    int pict_two[6]={0,0,W,H,0,0};
    int* ret=NULL;
    int Mx=0;
    int My=0;
    double tmp_nspr=0;
    int count=0;
    int effectives=0;
    int left,top=0;
    int right=BOUNDX-1;
    int down=BOUNDY-1;
    double dens=0;
    StructPointer p =(StructPointer)malloc(sizeof(StructPointerTest));
    
    //可以使用链表操作
    for(int i=0;i<BOUNDX;i++){
        if(i%SKIP==0){
            for(int j=0;j<BOUNDY;j++){
                if(j%SKIP==0){
                    pict1[0]=i;
                    pict1[1]=j;
                    ret=Diamond_Search(picture1,pict1,picture2);
                    total_num++;
                    sum_x+=(ret[0]-i);
                    sum_y+=(ret[1]-j);
                }
            }
        }
    }
    Mx=(sum_x/total_num);
    My=(sum_y/total_num);

    for(int i=0;i<BOUNDX;i++){
        if(i%SKIP==0){
            for(int j=0;j<BOUNDY;j++){
                if(j%SKIP==0){
                    pict_one[0]=i;
                    pict_one[1]=j;
                    pict_two[0]=i+Mx;
                    pict_two[1]=j+My;
                    if(PSNR(picture1,picture2,pict_one,pict_two)>tao){
                        result[count][0]=i;
                        result[count][1]=j;
                        result[count][2]=W;
                        result[count][3]=H;
                        result[count][4]=Mx;
                        result[count][5]=My;
                        effectives++;
                        count++;
                    }
                }
            }
        }
    }
    printf("effectives:%d\n",effectives);
    //while(dens<RATE && left<(BOUNDX-1)){
    //printf("left:%d\n",left);
    left=0;
    printf("BOUNDX:%d\n",BOUNDX);
    while(dens<RATE ){
        dens=density(result,left,top,right,down);
        left++;
        //printf("1 dens:%f\n",dens);
        //printf("now left:%d\n",left);
        if(left>=BOUNDX){
            printf("breaking!\n");
            break;
        }
    }
        left--;
    if(dens>=RATE){
        printf("1 dens:%f\n",dens);
        p->numx=left;
        p->numy=top;
        p->width=(right-left+1)*W;
        p->height=(down-top+1)*H;
        p->Mx=Mx;
        p->My=My;
        return p;
    }

    left=0;
    top=0;
    right=BOUNDX-1;
    down=BOUNDY-1;
    while(dens<RATE){
        dens=density(result,left,top,right,down);
        top++;
        //printf("2 dens:%f\n",dens);
        if(top>=BOUNDY){
            break;
        }
    }
    top--;
    if(dens>=RATE){
        printf("2 dens:%f\n",dens);
        p->numx=left;
        p->numy=top;
        p->width=(right-left+1)*W;
        p->height=(down-top+1)*H;
        p->Mx=Mx;
        p->My=My;
        return p;
    }

    left=0;
    top=0;
    right=BOUNDX-1;
    down=BOUNDY-1;
    while(dens<RATE && right>0){
        dens=density(result,left,top,right,down);
        right--;
        //printf("3 dens:%f\n",dens);
        if(right<0){
            break;
        }
    }
    right++;
    if(dens>=RATE){
        printf("3 dens:%f\n",dens);
        p->numx=left;
        p->numy=top;
        p->width=(right-left+1)*W;
        p->height=(down-top+1)*H;
        p->Mx=Mx;
        p->My=My;
        return p;
    }

    left=0;
    top=0;
    right=BOUNDX-1;
    down=BOUNDY-1;
    while(dens<RATE && down>0){
        dens=density(result,left,top,right,down);
        down--;
        //printf("4 dens:%f\n",dens);
        if(down<0){
            break;
        }
    }
    down++;
    if(dens>=RATE){
        printf("4 dens:%f\n",dens);
        p->numx=left;
        p->numy=top;
        p->width=(right-left+1)*W;
        p->height=(down-top+1)*H;
        p->Mx=Mx;
        p->My=My;
        return p;
    }
    printf("4 dens:%f\n",dens);
    p->numx=left;
    p->numy=top;
    p->width=(right-left+1)*W;
    p->height=(down-top+1)*H;
    p->Mx=Mx;
    p->My=My;
    return p;

    /* 
    p->numx=0;
    p->numy=0;
    p->width=240;
    p->height=H;
    p->Mx=Mx;
    p->My=My;
    return p;
    */

}
