#include "kernels/complex.c"

#define N 5
#define FLT_EPSILON 0x1.0p-23f

float Jacobi_am(float u, char arg, float x)
{
   float a[N+1];
   float g[N+1];
   float c[N+1];
   float two_n;
   float phi;
   float k;
   int n;

                        // Check special case x = 0 //
                        // i.e. k = m = alpha = 0.  //

   if ( x == 0.0 ) return u;

   switch (arg) {
      case 'a': k = sin( fabs((float) x) ); break;
      case 'm': k = sqrt( fabs((float) x) ); break;
      default:  k = (float) fabs(x);
   }

                   // Check special case k = 1 //

   if ( k == 1.0 ) return 2.0 * atan( exp(u) ) - M_PI_2;

         // If k > 1, then perform a Jacobi modulus transformation. //
         // Initialize the sequence of arithmetic and geometric     //
         // means, a = 1, g = k'.                                   //

   a[0] = 1.0f;
   g[0] = sqrt(1.0f - k * k);
   c[0] = k;
   
   // Perform the sequence of Gaussian transformations of arithmetic and //
   // geometric means of successive arithmetic and geometric means until //
   // the two means converge to a common mean (upto machine accuracy)    //
   // starting with a = 1 and g = k', which were set above.              //
   
   two_n = 1.0f; 
   for (n = 0; n < N; n++) {
      if ( fabs(a[n] - g[n]) < (a[n] * FLT_EPSILON) ) break;
      two_n += two_n;
      a[n+1] = 0.5f * (a[n] + g[n]);
      g[n+1] = sqrt(a[n] * g[n]);
      c[n+1] = 0.5f * (a[n] - g[n]);
   }

         // Prepare for the inverse transformation of phi = x * cm. //

   phi = two_n * a[n] * u;

                      // Perform backward substitution //

   for (; n > 0; n--) phi = 0.5L * ( phi + asin( c[n] * sin(phi) / a[n]) );

   return phi; 
}

float Jacobi_cn(float u, char arg,  float x) {
   return cos( Jacobi_am(u, arg, x) );
}

float Jacobi_sn(float u, char arg,  float x) {
   return sin( Jacobi_am(u, arg, x) );
}

float Jacobi_dn(float u, char arg,  float x) {
   float sn = sin(Jacobi_am(u, arg, x));

   switch (arg) {
      case 'm': return sqrt(1.0f - x * sn * sn);
      case 'a': x = sin( x );
      default:  x *= sn;
   }
   return sqrt(1.0f - x * x );
}

cfloat complex_cn(cfloat u, float m) {
   float cn_x = Jacobi_cn(real(u), 'm', m);
   float cn_y_c = Jacobi_cn(imag(u), 'm', 1-m);

   float sn_x = Jacobi_sn(real(u), 'm', m);
   float sn_y_c = Jacobi_sn(imag(u), 'm', 1-m);
   float dn_x = Jacobi_dn(real(u), 'm', m);
   float dn_y_c = Jacobi_dn(imag(u), 'm', 1-m);

   float common_den = cn_y_c*cn_y_c + m*sn_x*sn_x * sn_y_c*sn_y_c;
   cfloat complex_cn = (cfloat) (cn_x * cn_y_c, -sn_x*dn_x*sn_y_c*dn_y_c);
   complex_cn /= common_den;

   return complex_cn;
}