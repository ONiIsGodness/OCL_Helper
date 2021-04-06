__kernel void add_vector(
                         __global const float* a,
                         float b,
                         __global float* result){
    int gid = get_global_id(0);
    result[gid] = a[gid] + b;
}
