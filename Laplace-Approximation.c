/****
 * MPI Version of SOR solver algorithm for Laplace Approximation
 * Task-2
 ****/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>		

#define MASTER 1		/* Messaage Tags */
#define SLAVE 2
#define MSGINIT 0      
#define MAX_SIZE 4096	/* Maxium allowed Size of the Matrix */ 

int mtype;            	/* MPI Message Tag Type */
int N;					/* Given Size of the Matrix */ 
int maxnum;				/* Maximum number allowed in the Matrix Initialization */
char *Init;             /* Matrix initialization type */
double difflimit; 		/* Stop Condition */
double w;				/* Relaxation Factor */
int PRINT;				/* To View Output */

static double A[MAX_SIZE+2][MAX_SIZE+2]; /* (+2) - boundary elements */

MPI_Status status;

/* forward declarations */
int work(int, int);
int singlework();
void Init_Matrix(int);
void Print_Matrix();
void Init_Default();
int Read_Options(int, char **);

int 
main(int argc, char **argv)
{
    int iter, prank, np;
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Comm_rank(MPI_COMM_WORLD, &prank);
	Init_Default();					/* Init default values	*/
    Read_Options(argc,argv);		/* Read arguments	*/
	
	if(np == 1)						/* Only Master node will work if number of nodes are 1. */
	{
		if(prank == 0)
		{
			Init_Matrix(prank);
			iter = singlework();
			
			if(PRINT == 1){
				printf("\nInput Matrix processed by %d node\n", np);
				Print_Matrix();
			}
			
			printf("\nNumber of iterations :%d\n", iter);
		}
	}

	if(np>1)						/* When the number of nodes are greater than 1, the work is distributed among them by master node */	
		{
		if(prank == 0)                  /* Master Node */
		{
			Init_Matrix(prank);  		/* The matrix is initialized in master node, and it is sent to other child nodes */
			iter = work(prank, np);
		
			if(PRINT == 1) {
				printf("\n Input Matrix Processed by : %d nodes\n", np);
				Print_Matrix();
			}
		
			printf("\nNumber of iterations = %d\n", iter);
		
		}
		else {
									/* Child or Slave nodes receive their matrix part and begin their work */ 
			work(prank, np); 
		}
	}

	MPI_Finalize();
}

int singlework()
{
	int m, n;
	int iteration = 0;
	int finished = 0;
	double maxi, sum, prevmax_even, prevmax_odd;
	prevmax_even = 0.0;
	prevmax_odd = 0.0;
	while(!finished)
	{
		if((iteration%2) == 0)
		{
			for(m = 1; m < N+1; m++)
				{
					for(n = 1; n < N+1; n++)
					{
						if(((m+n) % 2) == 0)
						{
							A[m][n] = (1 - w) * A[m][n] + w *(A[m-1][n] + A[m+1][n] + A[m][n - 1] + A[m][n + 1])/4; 
						}
					}
				}
			
			
				/* Calculating the maximum sum of the elements by every processor node*/
				maxi = -999999.0;
				for (m = 1; m < N + 1; m++) {
					sum = 0.0;
					for (n = 1; n < N+1; n++) {
						sum += A[m][n];
					}
					if(sum > maxi)
						maxi = sum;
				}
			
				if(fabs(maxi - prevmax_even) <= difflimit)
					finished = 1;
				
				prevmax_even = maxi;
			
		}
		
		if((iteration%2) == 1)
		{
			for(m = 1; m < N+1; m++)
				{
					for(n = 1; n < N+1; n++)
					{
						if(((m+n) % 2) == 1)
						{
							A[m][n] = (1 - w) * A[m][n] + w *(A[m-1][n] + A[m+1][n] + A[m][n - 1] + A[m][n + 1])/4; 
						}
					}
				}
			
			
				/* Calculating the maximum sum of the elements by every processor node*/
				maxi = -999999.0;
				for (m = 1; m < N + 1; m++) {
					sum = 0.0;
					for (n = 1; n < N+1; n++) {
						sum += A[m][n];
					}
					if(sum > maxi)
						maxi = sum;
				}
			
				if(fabs(maxi - prevmax_odd) <= difflimit)
					finished = 1;
				
				prevmax_odd = maxi;
		}
		
		iteration++;
					
		if (iteration > 100000) 
		{
			/* exit if we don't converge fast enough */
			printf("Max number of iterations reached! Exit!\n");
			finished = 1;
		}
	}
	
	return iteration;
	
}



int
work(int rank, int p)
{
		int cols = N + 2;       			/* Number of columns to process by each node */
		int offset = N/p;					/* Number of rows to perform laplace approximation */
		int noffset = offset;				/* To send parts of matrix to child nodes */ 
		int rows = offset + 2;				/* Number of rows required to perform laplace approximation */
		int iteration = 0;					/* Number of iterations */
		int finished = 0;					/* To terminate the process */
		int m, n, i, j, x, y;				/* Looping variables */
		int dest, src;						/* Looping variables */
		double maxi;						/* Variables to calculate the max sum of elements */
		double sum;
		double prevmax_even, prevmax_odd;
		prevmax_even = 0.0;
		prevmax_odd = 0.0;
		
		if(rank == 0)		/* Sending individual parts to child nodes */
		{
			for(dest = 1; dest < p; dest++)
			{
				for(i = 0; i < rows; i++)
				{
					MPI_Send(&A[noffset + i][0], cols, MPI_DOUBLE, dest, MSGINIT, MPI_COMM_WORLD);
				}
				noffset += offset;
			}				
		}
		
		if(rank != 0)		/* Each child node recceives their corresponding part according to the noffset */
		{
			for(j = 0; j<rows; j++)
			{
				MPI_Recv(&A[j][0], cols, MPI_DOUBLE, 0, MSGINIT, MPI_COMM_WORLD, &status);
			}
		}
		
		
		while (!finished) 
		{
							/* Updating boundary rows */
			if (rank != 0)  /* Each node sends the top row shared with other nodes except master node */
			{	
				mtype = SLAVE;
				MPI_Send(&A[1][0], cols, MPI_DOUBLE, rank-1 ,mtype, MPI_COMM_WORLD);
				mtype = MASTER;
				MPI_Recv(&A[0][0], cols, MPI_DOUBLE, rank-1, mtype, MPI_COMM_WORLD, &status);
			}
	
			if (rank != p-1) 
			{				/* Each node sends the bottom row shared with other nodes */
				mtype = SLAVE;
				MPI_Recv(&A[offset+1][0], cols , MPI_DOUBLE, rank+1, mtype, MPI_COMM_WORLD, &status);
				mtype = MASTER;
				MPI_Send(&A[offset][0], cols, MPI_DOUBLE, rank+1, mtype, MPI_COMM_WORLD);
				
			}
			
			/* Calculating odd or even elements based on iteration */
			/* Calculate Part-A - Even elements */
			if((iteration%2) == 0)
			{
				for(m = 1; m < offset+1; m++)
				{
					for(n = 1; n < N+1; n++)
					{
						if(((m+n) % 2) == 0)
						{
							A[m][n] = (1 - w) * A[m][n] + w *(A[m-1][n] + A[m+1][n] + A[m][n - 1] + A[m][n + 1])/4; 
						}
					}
				}
			
			
				/* Calculating the maximum sum of the elements by every processor node*/
				maxi = -999999.0;
				for (m = 1; m < offset + 1; m++) {
					sum = 0.0;
					for (n = 1; n < N+1; n++) {
						sum += A[m][n];
					}
					if(sum > maxi)
						maxi = sum;
				}
			
				double maxsum1;
				/* Finding maximum sum across all the nodes */
				MPI_Allreduce(&maxi, &maxsum1, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
				/* Compare the sum with the prev sum, i.e., check whether we are done or not */
				if(fabs(maxsum1 - prevmax_even) <= difflimit)
					finished = 1;
				
				prevmax_even = maxsum1;
				
			}
			/* Calculate Part-B Odd elements */
			if ((iteration%2) == 1)
			{
				for(m = 1; m < offset+1; m++)
				{
					for(n = 1; n < N+1; n++)
					{
						if(((m+n) % 2) == 1)
						{
							A[m][n] = (1 - w) * A[m][n] + w *(A[m-1][n] + A[m+1][n] + A[m][n - 1] + A[m][n + 1])/4; 
						}
					}
				}
			
			
				/* Calculating the maximum sum of the elements by every processor node*/
				maxi = -999999.0;
				for (m = 1; m < offset + 1; m++) {
					sum = 0.0;
					for (n = 1; n < N+1; n++) {
						sum += A[m][n];
					}
					if(sum > maxi)
						maxi = sum;
				}
			
				double maxsum2;
				/* Finding maximum sum across all the nodes */
				MPI_Allreduce(&maxi, &maxsum2, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
				/* Compare the sum with the prev sum, i.e., check whether we are done or not */
				if(fabs(maxsum2 - prevmax_odd) <= difflimit)
					finished = 1;
				
				prevmax_odd = maxsum2;
				
			}
			
			iteration++;
					
			if (iteration > 100000) 
			{
				/* exit if we don't converge fast enough */
				printf("Max number of iterations reached! Exit!\n");
				finished = 1;
			}

		}
		
		/*Sending results back to the Master node processor */
		if (rank != 0)
		{
			mtype = SLAVE;
			for(x = 1; x < offset+1; x++)
			{
				MPI_Send(&A[x][0], N+2, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD);	
			}
		}
		
		/* Offset to locate the position of returning result rows in matrix A back in the master node */
		int recvoffset = offset+1;
		
		if(rank == 0)
		{
			mtype = SLAVE;
			for(src = 1; src < p; src++)
			{
				for(y = 0; y<offset; y++)
				{
					MPI_Recv(&A[recvoffset][0], N+2, MPI_DOUBLE, src, mtype, MPI_COMM_WORLD, &status);
				    recvoffset++;
				}
			}	
		}
		
	return iteration;
}
 
void
Init_Matrix(int rank)
{
    int i, j, dmmy;
	
	printf("\nsize      = %dx%d ",N,N);
	printf("\nmaxnum    = %d \n",maxnum);
	printf("difflimit = %.7lf \n",difflimit);
	printf("Init	  = %s \n",Init);
	printf("w	  = %f \n\n",w);
	printf("Initializing matrix...");
	
    /* Initialize all grid elements, including the boundary */
    for (i = 0; i < N+2; i++) {
	for (j = 0; j < N+2; j++) {
	    A[i][j] = 0.0;
	}
    }
    if (strcmp(Init,"count") == 0) {
	for (i = 1; i < N+1; i++){
	    for (j = 1; j < N+1; j++) {
			A[i][j] = (double)i/2;
	    }
	}
    }
    if (strcmp(Init,"rand") == 0) {
	for (i = 1; i < N+1; i++){
	    for (j = 1; j < N+1; j++) {
			A[i][j] = (rand() % maxnum) + 1.0;
	    }
	}
    }
    if (strcmp(Init,"fast") == 0) {
	for (i = 1; i < N+1; i++){
	    dmmy++;
	    for (j = 1; j < N+1; j++) {
		dmmy++;
		if ((dmmy%2) == 0)
		    A[i][j] = 1.0;
		else
		    A[i][j] = 5.0;
	    }
	}
    }

    /* Set the border to the same values as the outermost rows/columns */
    /* fix the corners */
    A[0][0] = A[1][1];
    A[0][N+1] = A[1][N];
    A[N+1][0] = A[N][1];
    A[N+1][N+1] = A[N][N];
    /* fix the top and bottom rows */
    for (i = 1; i < N+1; i++) {
	A[0][i] = A[1][i];
	A[N+1][i] = A[N][i];
    }
    /* fix the left and right columns */
    for (i = 1; i < N+1; i++) {
	A[i][0] = A[i][1];
	A[i][N+1] = A[i][N];
    }

    printf("done in node: %d \n", rank);
    if (PRINT == 1)
		Print_Matrix();
}

void
Print_Matrix()
{
    int i, j;
 
    for (i=0; i<N+2 ;i++){
	for (j=0; j<N+2 ;j++){
	    printf(" %f",A[i][j]);
	}
	printf("\n");
    }
    printf("\n");
}


void 
Init_Default()
{
    N = 2048;
    difflimit = 0.00001*N;
    Init = "rand";
    maxnum = 15.0;
    w = 0.5;
    PRINT = 0;
}

int
Read_Options(int argc, char **argv)
{
    char    *prog;
 
    prog = *argv;
    while (++argv, --argc > 0)
	if (**argv == '-')
	    switch ( *++*argv ) {
	    case 'n':
		--argc;
		N = atoi(*++argv);
		difflimit = 0.00001*N;
		break;
	    case 'h':
		printf("\nHELP: try sor -u \n\n");
		exit(0);
		break;
	    case 'u':
		printf("\nUsage: sor [-n problemsize]\n");
		printf("           [-d difflimit] 0.1-0.000001 \n");
		printf("           [-D] show default values \n");
		printf("           [-h] help \n");
		printf("           [-I init_type] fast/rand/count \n");
		printf("           [-m maxnum] max random no \n");
		printf("           [-P print_switch] 0/1 \n");
		printf("           [-w relaxation_factor] 1.0-0.1 \n\n");
		exit(0);
		break;
	    case 'D':
		printf("\nDefault:  n         = %d ", N);
		printf("\n          difflimit = %f ", difflimit);
		printf("\n          Init      = rand" );
		printf("\n          maxnum    = 5 ");
		printf("\n          w         = 0.5 \n");
		printf("\n          P         = 0 \n\n");
		exit(0);
		break;
	    case 'I':
		--argc;
		Init = *++argv;
		break;
	    case 'm':
		--argc;
		maxnum = atoi(*++argv);
		break;
	    case 'd':
		--argc;
		difflimit = atof(*++argv);
		break;
	    case 'w':
		--argc;
		w = atof(*++argv);
		break;
	    case 'P':
		--argc;
		PRINT = atoi(*++argv);
		break;
	    default:
		printf("%s: ignored option: -%s\n", prog, *argv);
		printf("HELP: try %s -u \n\n", prog);
		break;
	    } 
}

