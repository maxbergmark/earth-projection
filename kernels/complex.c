#pragma once

//2 component vector to hold the real and imaginary parts of a complex number:
typedef float2 cfloat;

#define I ((cfloat)(0.0, 1.0))

void cprint(cfloat a) {
    printf("%.5f + %.5fi", a.x, a.y);
}

/*
 * Return Real (Imaginary) component of complex number:
 */
inline float  real(cfloat a){
     return a.x;
}
inline float  imag(cfloat a){
     return a.y;
}

/*
 * Get the modulus of a complex number (its length):
 */
inline float cmod(cfloat a){
    return (sqrt(a.x*a.x + a.y*a.y));
}

/*
 * Get the argument of a complex number (its angle):
 * http://en.wikipedia.org/wiki/Complex_number#Absolute_value_and_argument
 */
inline float carg(cfloat a){
    // return atan2(a.x, a.y);
    if(a.x > 0){
        return atan(a.y / a.x);

    }else if(a.x < 0 && a.y >= 0){
        return atan(a.y / a.x) + M_PI;

    }else if(a.x < 0 && a.y < 0){
        return atan(a.y / a.x) - M_PI;

    }else if(a.x == 0 && a.y > 0){
        return M_PI/2;

    }else if(a.x == 0 && a.y < 0){
        return -M_PI/2;

    }else{
        return 0;
    }
}

/*
 * Multiply two complex numbers:
 *
 *  a = (aReal + I*aImag)
 *  b = (bReal + I*bImag)
 *  a * b = (aReal + I*aImag) * (bReal + I*bImag)
 *        = aReal*bReal +I*aReal*bImag +I*aImag*bReal +I^2*aImag*bImag
 *        = (aReal*bReal - aImag*bImag) + I*(aReal*bImag + aImag*bReal)
 */
inline cfloat  cmult(cfloat a, cfloat b){
    return (cfloat)( a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}


/*
 * Divide two complex numbers:
 *
 *  aReal + I*aImag     (aReal + I*aImag) * (bReal - I*bImag)
 * ----------------- = ---------------------------------------
 *  bReal + I*bImag     (bReal + I*bImag) * (bReal - I*bImag)
 * 
 *        aReal*bReal - I*aReal*bImag + I*aImag*bReal - I^2*aImag*bImag
 *     = ---------------------------------------------------------------
 *            bReal^2 - I*bReal*bImag + I*bImag*bReal  -I^2*bImag^2
 * 
 *        aReal*bReal + aImag*bImag         aImag*bReal - Real*bImag 
 *     = ---------------------------- + I* --------------------------
 *            bReal^2 + bImag^2                bReal^2 + bImag^2
 * 
 */
inline cfloat cdiv(cfloat a, cfloat b){
    return (cfloat)((a.x*b.x + a.y*b.y)/(b.x*b.x + b.y*b.y), 
        (a.y*b.x - a.x*b.y)/(b.x*b.x + b.y*b.y));
}


/*
 *  Square root of complex number.
 *  Although a complex number has two square roots, numerically we will
 *  only determine one of them -the principal square root, see wikipedia
 *  for more info: 
 *  http://en.wikipedia.org/wiki/Square_root#Principal_square_root_of_a_complex_number
 */
 inline cfloat csqrt(cfloat a){
     return (cfloat)(sqrt(cmod(a)) * cos(carg(a)/2), 
        sqrt(cmod(a)) * sin(carg(a)/2));
 }