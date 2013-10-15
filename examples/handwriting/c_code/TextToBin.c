#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void convMatrix(char* filename, int rows, int cols)
{
  int i, j;
  int** m;
  FILE* fp;
  char fn[64];
  int count = 0;
  strcpy(fn, "Input/");
  strcat(fn, filename);
  strcat(fn, ".txt");

  fp = fopen(fn, "r");
  m = malloc(rows * sizeof(float*));
  for (i = 0; i < rows; i++) {
    m[i] = malloc(sizeof(float) * cols);
    for (j = 0; j < cols; j++)
      fscanf(fp, "%i", &m[i][j]);
  }
  fclose(fp);

  printf("const float %s[] = {\n", filename);
  for (i = 0; i < rows; i++)
    for (j = 0; j < cols; j++) {
      printf("%i,", m[i][j]);
      if (count == 7) printf("\n");
      count = (count+1) % 8;
    }
  printf("0};\n\n");
}


int main(int argc, char *argv[])
{
  convMatrix("mat_1_b", 1, 1000);
  convMatrix("mat_1_w", 784, 1000);
  convMatrix("mat_2_b", 1, 500);
  convMatrix("mat_2_w", 1000, 500);
  convMatrix("mat_3_b", 1, 300);
  convMatrix("mat_3_w", 500, 300);
  convMatrix("mat_4_b", 1, 50);
  convMatrix("mat_4_w", 300, 50);
  convMatrix("SemPtr", 10, 50);
  convMatrix("samplesPtr", 100, 784);
  return 0;
}
