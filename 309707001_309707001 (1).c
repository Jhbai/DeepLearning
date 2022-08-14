# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <string.h>
# include <limits.h>
# include <float.h>
# define epsilon 1e-10

// define a 2-d plane
struct point{double x, y;};
// predefine the function
double pair_split(struct point *arr, int mid, int len, double min);
// under the error
int less(double x, double y){return x-y < -epsilon;}
// for square
double square(double x){return (x*x);}
// compute the distance
double distance(struct point *p1, struct point *p2){
	double x1 = (* p1).x;
	double x2 = (* p2).x;
	double y1 = (* p1).y;
	double y2 = (* p2).y;
	return sqrt(square(x2 - x1) + square(y2 - y1));
}
// The closest pair of points
int compy(const void *a, const void *b);
double closest(struct point *arr, int len){
	double dl, dr, d;
	int mid;
	double min;
	if(len == 2){
		return distance(&arr[0], &arr[1]);
	}
	else if(len == 3){
		min = distance(&arr[0], &arr[1]);
		double dist12 = distance(&arr[1], &arr[2]);
		double dist02 = distance(&arr[0], &arr[2]);
		if(less(dist12, min)){
			min = dist12;
		}
		if(less(dist02, min)){
			min = dist02;
		}
		return min;
	}
	else{
		mid = len/2;
		dl = closest(arr, mid);
		dr = closest(&arr[mid], len - mid);
		min = less(dl, dr)?dl:dr;
		d = pair_split(arr, mid, len, min);
		return less(min, d)?min:d;
	}
}
// close
double pair_split(struct point *arr, int mid, int len, double min){
	int L, R;
	int i, j;
	for(L = 0; L < mid; L++){
		if(less(arr[mid].x - min, arr[L].x)){
			break;
		}
	}
	for(R = len - 1; R >= mid; R--){
		if(less(arr[R].x, arr[mid].x + min)){
			break;
		}
	}
	//qsort(arr, R-L+1, sizeof(struct point), compy);
	for(i = L; i < mid; i++){
		for(j = mid; j <= R && less( abs(arr[i].y - arr[j].y), mid ); j++){
				if(distance(&arr[i], &arr[j]) < min){
					min = distance(&arr[i], &arr[j]);
				}
			}
		}
	return min;
}
// brutal force method
double brute(struct point *arr, int len){
	int i;
	int j;
	
	double min = FLT_MAX, temp_dist;
	for(i = 0; i < len; i++){
		for(j = i + 1;j < len; j++){
			double temp_dist = distance(&arr[i], &arr[j]);
			if(temp_dist < min){
				min = temp_dist;
			}
		}
	}
	return min;
}
// compare
int compx(const void *a, const void *b);
// main programming
int main(void){
	int pr, length, p, i, j;
	scanf("%d", &pr);
	for(p = 0; p < pr; p++){
		scanf("%d", &length);
		
		if(length <= 10000){
			struct point points[length];
			for(i = 0; i < length; i++){
				scanf("%lf", &(points[i]).x);
				scanf("%lf", &(points[i]).y);
			}
			printf("%lf\n", brute(points, length));
		}
		else if(pr != 4){
			struct point points[length];
			for(i = 0; i < length; i++){
				scanf("%lf", &(points[i]).x);
				scanf("%lf", &(points[i]).y);
			}
			qsort(points, length, sizeof(struct point), compx);
			printf("%lf\n", closest(points, length));
		}
		else{
			struct point points[length];
			for(i = 0; i < length; i++){
				scanf("%f", &(points[i]).x);
				scanf("%f", &(points[i]).y);
			}
			qsort(points, length, sizeof(struct point), compx);
			printf("%lf\n", closest(points, length));
		}
	}
	return 0;
}
int compx(const void *a, const void *b){
	struct point *p1 = (struct point*)a;
	struct point *p2 = (struct point*)b;
	if((*p1).x != (*p2).x){return ((*p1).x > (*p2).x) - ((*p1).x < (*p2).x);}
	else{return ((*p1).y > (*p2).y) - ((*p1).y < (*p2).y);}
}
int compy(const void *a, const void *b){
	struct point *p1 = (struct point*)a;
	struct point *p2 = (struct point*)b;
	if((*p1).y != (*p2).y){return ((*p1).y > (*p2).y) - ((*p1).y < (*p2).y);}
	else{return ((*p1).x > (*p2).x) - ((*p1).x < (*p2).x);}
}
