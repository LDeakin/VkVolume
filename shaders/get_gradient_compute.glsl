#if defined(GRADIENT_MAP_SET) && defined(GRADIENT_MAP_BINDING)
layout (set = GRADIENT_MAP_SET, binding = GRADIENT_MAP_BINDING, r8) uniform image3D gradient_map;
#endif

float get_gradient(ivec3 pos, ivec3 dim1) {
  if (!transfer_function_uniform.use_gradient) {
    return 1.0f;
  } else {
#ifdef PRECOMPUTED_GRADIENT
    return imageLoad(gradient_map, pos).x;
#else
    // Gradient on-the-fly using tetrahedron technique http://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
    ivec2 k = ivec2(1,-1);
    vec3 gradientDir = 0.25f * (
      k.xyy * imageLoad(volume, clamp(pos + k.xyy, ivec3(0), dim1)).x +
      k.yyx * imageLoad(volume, clamp(pos + k.yyx, ivec3(0), dim1)).x +
      k.yxy * imageLoad(volume, clamp(pos + k.yxy, ivec3(0), dim1)).x +
      k.xxx * imageLoad(volume, clamp(pos + k.xxx, ivec3(0), dim1)).x);
    float gradient = clamp(length(gradientDir) * transfer_function_uniform.grad_magnitude_modifier, 0, 1);
    return gradient;
#endif
  }
}