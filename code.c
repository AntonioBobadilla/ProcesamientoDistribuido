#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <malloc.h>
#include "omp.h"
#include <string.h>

// Standard values located at the header of an BMP file
#define MAGIC_VALUE    0X4D42 
//Bit depth
#define BITS_PER_PIXEL 24
#define NUM_PLANE      1
#define COMPRESSION    0
#define BITS_PER_BYTE  8
//OpenMP var
#define NUM_THREADS 15

#pragma pack(1)

/*Section used to declare structures*/
typedef struct{
  uint16_t type;
  uint32_t size;
  uint16_t reserved1;
  uint16_t reserved2;
  uint32_t offset;
  uint32_t header_size;
  uint32_t width;
  uint32_t height;
  uint16_t planes;
  uint16_t bits;
  uint32_t compression;
  uint32_t imagesize;
  uint32_t xresolution;
  uint32_t yresolution;
  uint32_t importantcolours;
}BMP_Header;

typedef struct{
  BMP_Header header;
  unsigned int pixel_size;
  unsigned int width;
  unsigned int height;
  unsigned int bytes_per_pixel;
  unsigned char * pixel; //For future allocation in memory
}BMP_Image;

/*Section used to declare functions*/
int checkHeader(BMP_Header *);
BMP_Image* cleanUp(FILE *, BMP_Image *);
BMP_Image* BMP_open(const char *);
int BMP_save(const BMP_Image *img, const char *filename);
void BMP_destroy(BMP_Image *img);
static int RGB2Gray(unsigned char, unsigned char, unsigned char);
void BMP_gray(BMP_Image*);

/*End section*/

int checkHeader(BMP_Header *hdr){
  if((hdr->type) != MAGIC_VALUE) {
    printf("No es un bmp\n");
    return 0;
  }
  if((hdr->bits) != BITS_PER_PIXEL) {
    printf("Revisa bit depth\n");
    return 0;
  }
  if((hdr->planes) != NUM_PLANE) {
    printf("Array de diferente dimensiones\n");
    return 0;
  }
  if((hdr->compression) != COMPRESSION) {
    printf("Hay compresion\n");
    return 0;
  }
  return 1;
}

BMP_Image* cleanUp(FILE *fptr, BMP_Image *img) {
  if(fptr != NULL) {
    fclose(fptr);
  }
  if(img != NULL) {
    if(img->pixel != NULL) {
      free(img->pixel);
    }
    free(img);
  }
  return NULL;
}


BMP_Image* BMP_open(const char *filename){
  FILE *fptr = NULL;
  BMP_Image *img = NULL;

  // Abrir el archivo
  fptr = fopen(filename, "rb");
  if(fptr == NULL) {
    printf("Archivo no existe\n");
    return cleanUp(fptr, img);
  }

  // Asignar memoria para la estructura BMP_Image
  img = malloc(sizeof(BMP_Image));
  if(img == NULL) {
    return cleanUp(fptr, img);
  }

  // Leer el encabezado BMP_Header del archivo
  if(fread(&(img -> header), sizeof(BMP_Header), 1, fptr) != 1) {
    printf("Header no disponible\n");
    return cleanUp(fptr, img);
  }

  // Verificar si el encabezado es válido
  if(checkHeader(&(img -> header)) == 0) {
    printf("Header fuera del estandar\n");
    return cleanUp(fptr, img);
  }

  // Asignar los valores de la estructura BMP_Image
  img -> pixel_size      = (img -> header).size - sizeof(BMP_Header);
  img -> width           = (img -> header).width;
  img -> height          = (img -> header).height;
  img -> bytes_per_pixel = (img -> header).bits / BITS_PER_BYTE;

  // Asignar memoria para los datos de píxeles
  img -> pixel = malloc(sizeof(unsigned char) * (img -> pixel_size));
  if((img -> pixel) == NULL) {
    printf("Imagen vacia\n");
    return cleanUp(fptr, img);
  }

  // Leer los datos de píxeles del archivo
  if(fread(img->pixel, sizeof(char), img -> pixel_size, fptr) != (img -> pixel_size)) {
    printf("Imagen con contenido irregular\n");
    return cleanUp(fptr, img);
  }

  // Verificar si hay bytes residuales después de leer los píxeles
  char onebyte;
  if(fread(&onebyte, sizeof(char), 1, fptr) != 0) {
    printf("Hay pixeles residuales\n");
    return cleanUp(fptr, img);
  }

  // Cerrar el archivo y retornar la estructura BMP_Image
  fclose(fptr);
  return img;
}


int BMP_save(const BMP_Image *img, const char *filename){
  FILE *fptr = NULL;
  fptr = fopen(filename, "wb");
  if(fptr == NULL) {
    return 0;
  }
  if(fwrite(&(img->header), sizeof(BMP_Header), 1, fptr) != 1) {
    fclose(fptr);
    return 0;
  }
  if(fwrite(img->pixel, sizeof(char), img->pixel_size, fptr) != img->pixel_size) {
    fclose(fptr);
    return 0;
  }
  fclose(fptr);
  return 1;
}

void BMP_destroy(BMP_Image *img){
  free (img -> pixel);
  free (img);
}


static int  RGB2Gray(unsigned char red, unsigned char green, unsigned char blue){
  double gray =  0.2989*red + 0.5870*green + 0.1140*blue;
  return (int) gray;
}

void BMP_gray(BMP_Image *img){
  omp_set_num_threads(NUM_THREADS);
  const double startTime = omp_get_wtime();
  
  #pragma omp parallel
  {
    #pragma omp for // No wait
    for (int pxl = 0; pxl < (img->pixel_size); pxl += 3)
    {
      unsigned char gray = RGB2Gray(img->pixel[pxl + 2], img->pixel[pxl + 1], img->pixel[pxl]);
      img->pixel[pxl + 2] = gray; // Red pixel
      img->pixel[pxl + 1] = gray; // Green pixel
      img->pixel[pxl]     = gray; // Blue pixel
    }
  }
  
  if (BMP_save(img, "GrayScale.bmp") == 0)
  {
    printf("Output file invalid!\n");
    BMP_destroy(img);
  }
  
  // Destroy the BMP image
  BMP_destroy(img);
  const double endTime = omp_get_wtime();
  printf("En un tiempo total de (%lf)\n", (endTime - startTime));
}

/*Debugging functions*/
float ** kernel(unsigned int size){
  unsigned int height = size;
  unsigned int width = size * 3; 
  float ** matrix = malloc(sizeof(float*) * height);
  if (matrix == NULL) {
    printf("No se pudo asignar memoria\n");
    return NULL;
  }

  for(int i = 0; i < height; i++){
    matrix[i] = malloc(sizeof(float) * width);
    if (matrix[i] == NULL) {
      printf("No se pudo asignar memoria\n");
      for (int j = 0; j < i; j++) {
        free(matrix[j]);
      }
      free(matrix);
      return NULL;
    }
  }
  
  float value = 1.0 / (size * size);
  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      matrix[i][j] = value;
    }
  }
  
  return matrix;
}

char** pixelMat(BMP_Image* img) {
  const unsigned int height = img->height;
  const unsigned int width = img->width * 3;
  
  // Asignamos memoria contigua para todo el contenido de la matriz
  char* matData = malloc(sizeof(char) * height * width);
  
  #pragma omp parallel for schedule(dynamic, NUM_THREADS)
  for (unsigned int i = 0; i < height; i++) {
    // Apuntamos a la dirección de memoria de la fila i-ésima en la matriz
    char* row = &matData[i * width];
    // Copiamos los valores de los píxeles de la imagen a la fila correspondiente
    memcpy(row, &img->pixel[i * img->width * 3], width * sizeof(char));
  }
  
  // Asignamos memoria para el arreglo de apuntadores a las filas
  char** mat = malloc(sizeof(char*) * height);
  for (unsigned int i = 0; i < height; i++) {
    // Apuntamos a la dirección de memoria de la fila i-ésima en la matriz
    mat[i] = &matData[i * width];
  }
  
  return mat;
}

void BMP_blur(char* filename, unsigned int size){
  BMP_Image* img = BMP_open(filename);
  char** out_buffer = pixelMat(img);
  float** Createkernel = kernel(size);

  unsigned int height = img->height;
  unsigned int width = img->width * 3;
  int M = (size-1)/2;

  omp_set_num_threads(NUM_THREADS);
  
  #pragma omp parallel for collapse(2) schedule(dynamic)
  for(int i = M; i < height-M; i++){
    for(int j = M; j < width-M; j++){
      float sum = 0.0;
      for(int k = -M; k <= M; k++){
        for(int l = -M; l <= M; l++){
          sum += Createkernel[k+M][l+M] * img->pixel[(i+k)*width+(j+l)];
        }
      }
      out_buffer[i][j] = (char)sum;
    }
  }

  #pragma omp parallel for collapse(2) schedule(dynamic)
  for(int i = 1; i < height-1; i++){
    for(int j = 1; j < width-1; j++){
      img->pixel[i*width+j] = out_buffer[i][j];
    }
  }

  char name[20];
  if(strcmp(filename,"HorizontalRot.bmp") == 0){
    sprintf(name, "Rblur%02dX.bmp", size);
  } else {
    sprintf(name, "Blur%02d.bmp", size);
  }

  //Guardar la imagen
  if (BMP_save(img, name) == 0){
    printf("Error al guardar la imagen!\n");
    BMP_destroy(img);
    free(Createkernel);
    free(out_buffer);
    return;
  }

  // Liberar memoria
  BMP_destroy(img);
  free(Createkernel);
  free(out_buffer);
}

int main(){
  omp_set_num_threads(NUM_THREADS);

  #pragma omp sections
{
    /*#pragma omp section
    BMP_blur("original.bmp",3);
    #pragma omp section
    BMP_blur("original.bmp",5);
    #pragma omp section
    BMP_blur("original.bmp",7);
    #pragma omp section
    BMP_blur("original.bmp",9);
    #pragma omp section
    BMP_blur("original.bmp",11);
    #pragma omp section
    BMP_blur("original.bmp",13);
    #pragma omp section
    BMP_blur("original.bmp",15);
    #pragma omp section
    BMP_blur("original.bmp",17);
    #pragma omp section
    BMP_blur("original.bmp",19);
    #pragma omp section
    BMP_blur("original.bmp",21);
    #pragma omp section
    BMP_blur("original.bmp",23);
    #pragma omp section
    BMP_blur("original.bmp",25);
    #pragma omp section
    BMP_blur("original.bmp",27);
    #pragma omp section
    BMP_blur("original.bmp",29);
    #pragma omp section
    BMP_blur("original.bmp",31);
    #pragma omp section
    BMP_blur("original.bmp",33);
    #pragma omp section
    BMP_blur("original.bmp",35);
    #pragma omp section
    BMP_blur("original.bmp",37);
    #pragma omp section
    BMP_blur("original.bmp",39);
    #pragma omp section
    BMP_blur("original.bmp",41);
    #pragma omp section
    BMP_blur("original.bmp",43);
    #pragma omp section
    BMP_blur("original.bmp",45);
    #pragma omp section
    BMP_blur("original.bmp",47);
    #pragma omp section
    BMP_blur("original.bmp",49);
    #pragma omp section
    BMP_blur("original.bmp",51);
    #pragma omp section
    BMP_blur("original.bmp",53);
    #pragma omp section
    BMP_blur("original.bmp",55);
    #pragma omp section
    BMP_blur("original.bmp",57);
    #pragma omp section
    BMP_blur("original.bmp",59);
    #pragma omp section
    BMP_blur("original.bmp",61);
    #pragma omp section
    BMP_blur("original.bmp",63);
    #pragma omp section
    BMP_blur("original.bmp",65);*/
    #pragma omp section
    BMP_blur("original.bmp",67);
    #pragma omp section
    BMP_blur("original.bmp",69);
    #pragma omp section
    BMP_blur("original.bmp",71);
    #pragma omp section
    BMP_blur("original.bmp",73);
    #pragma omp section
    BMP_blur("original.bmp",75);
    #pragma omp section
    BMP_blur("original.bmp",77);
    #pragma omp section
    BMP_blur("original.bmp",79);
    #pragma omp section
    BMP_blur("original.bmp",81);

  }

  return 0;
}