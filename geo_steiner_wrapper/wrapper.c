// wrapper.c
#include "geosteiner.h"


// compile with
// gcc -fPIC -shared wrapper_so.c -o libwrapper.so  ~/programs/geosteiner-5.3/libgeosteiner.a -lgmp -lm

// libtool --mode=link gcc -fPIC -shared wrapper.c -o libwrapper.la libgeosteiner.la -lgmp -lm



// Direct passthrough to gst_open_geosteiner()
int wrapper_open_geosteiner() {
    return gst_open_geosteiner();
}

// Wrapper for gst_esmt
int wrapper_esmt(int num_points,
                 const double *points,
                 double *length,
                 int *num_steiner,
                 double *steiner_coords,
                 int *num_edges,
                 int *edges) {
    return gst_esmt(num_points, points, length, num_steiner, steiner_coords, num_edges, edges, NULL, NULL);
}

// Wrapper for gst_rsmt
int wrapper_rsmt(int num_points,
                 const double *points,
                 double *length,
                 int *num_steiner,
                 double *steiner_coords,
                 int *num_edges,
                 int *edges) {
    return gst_rsmt(num_points, points, length, num_steiner, steiner_coords, num_edges, edges, NULL, NULL);
}
