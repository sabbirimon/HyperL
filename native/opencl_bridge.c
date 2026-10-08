/* Original HyperL OpenCL host bridge. Apache-2.0. GPU only; no silent CPU fallback. */
#define CL_TARGET_OPENCL_VERSION 120
#include "include/hyperl_contract.h"
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <errno.h>

static cl_device_id devices[128];
static cl_uint count;
static int collect(void) {
    cl_platform_id platforms[32];
    cl_uint n = 0;
    cl_int e = clGetPlatformIDs(32, platforms, &n);
    if (e != CL_SUCCESS || n > 32)
        return 0;
    for (cl_uint i = 0; i < n; i++) {
        cl_device_id found[128];
        cl_uint size = 0;
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 128, found, &size) != CL_SUCCESS)
            continue;
        if (size > 128 || count + size > 128)
            return 0;
        for (cl_uint j = 0; j < size; j++)
            devices[count++] = found[j];
    }
    return count > 0;
}
static void json_string(const char *s) {
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (*p == '"' || *p == '\\') {
            putchar('\\');
            putchar(*p);
        } else if (*p < 32)
            printf("\\u%04x", *p);
        else
            putchar(*p);
    }
    putchar('"');
}
static void info(cl_device_id d, cl_device_info field) {
    char text[512] = {0};
    if (clGetDeviceInfo(d, field, sizeof(text), text, NULL) != CL_SUCCESS)
        strcpy(text, "unknown");
    text[511] = 0;
    json_string(text);
}
static long number(const char *s, long max) {
    char *end;
    errno = 0;
    long n = strtol(s, &end, 10);
    return errno || !*s || *end || n < 0 || n > max ? -1 : n;
}
static void *read_exact(const char *file, size_t size) {
    FILE *f = fopen(file, "rb");
    if (!f)
        return NULL;
    void *b = malloc(size);
    if (!b) {
        fclose(f);
        return NULL;
    }
    if (fread(b, 1, size, f) != size || fgetc(f) != EOF || ferror(f)) {
        free(b);
        fclose(f);
        return NULL;
    }
    fclose(f);
    return b;
}
static int filename(char *out, size_t limit, const char *directory, const char *name) {
    int n = snprintf(out, limit, "%s/%s", directory, name);
    return n > 0 && (size_t)n < limit;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--probe")) {
        int available = collect();
        printf("{\"backend\":\"OpenCL "
               "GPU\",\"inferenceQualified\":false,\"available\":%s,\"devices\":[",
               available ? "true" : "false");
        for (cl_uint i = 0; i < count; i++) {
            if (i)
                putchar(',');
            printf("{\"index\":%u,\"name\":", i);
            info(devices[i], CL_DEVICE_NAME);
            printf(",\"vendor\":");
            info(devices[i], CL_DEVICE_VENDOR);
            printf(",\"driver\":");
            info(devices[i], CL_DRIVER_VERSION);
            printf(",\"runtime\":");
            info(devices[i], CL_DEVICE_VERSION);
            printf("}");
        }
        puts("]}");
        return 0;
    }
    if (argc != 6 || strcmp(argv[1], "--run")) {
        fputs("Use --probe or --run DEVICE N INPUT_COUNT DIRECTORY\n", stderr);
        return 2;
    }
    long selected = number(argv[2], 127), length = number(argv[3], HL_MAX_VECTOR_ELEMENTS),
         inputs = number(argv[4], HL_MAX_INPUTS);
    if (selected < 0 || length < 1 || inputs < 1 || !collect() || selected >= (long)count) {
        fputs("GPU or bounded arguments unavailable\n", stderr);
        return 3;
    }
    cl_device_id device = devices[selected];
    cl_bool endian = CL_FALSE;
    uint32_t word = 1;
    if (*(unsigned char *)&word != 1 ||
        clGetDeviceInfo(device, CL_DEVICE_ENDIAN_LITTLE, sizeof(endian), &endian, NULL) !=
            CL_SUCCESS ||
        !endian) {
        fputs("Little-endian host/device required\n", stderr);
        return 3;
    }
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_mem buffers[HL_MAX_INPUTS + 1] = {0};
    float *result = NULL;
    char *source = NULL;
    cl_int error;
    int status = 4;
    char file[4096];
    size_t bytes = (size_t)length * sizeof(float);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
    if (error != CL_SUCCESS)
        goto done;
    queue = clCreateCommandQueue(context, device, 0, &error);
    if (error != CL_SUCCESS)
        goto done;
    if (!filename(file, sizeof(file), argv[5], "kernel.cl"))
        goto done;
    FILE *f = fopen(file, "rb");
    if (!f)
        goto done;
    if (fseek(f, 0, SEEK_END)) {
        fclose(f);
        goto done;
    }
    long source_length = ftell(f);
    if (source_length < 1 || source_length > 65536 || fseek(f, 0, SEEK_SET)) {
        fclose(f);
        goto done;
    }
    source = malloc((size_t)source_length + 1);
    if (!source) {
        fclose(f);
        goto done;
    }
    if (fread(source, 1, (size_t)source_length, f) != (size_t)source_length) {
        fclose(f);
        goto done;
    }
    fclose(f);
    source[source_length] = 0;
    const char *src = source;
    size_t size = (size_t)source_length;
    program = clCreateProgramWithSource(context, 1, &src, &size, &error);
    if (error != CL_SUCCESS)
        goto done;
    error = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", NULL, NULL);
    if (error != CL_SUCCESS) {
        char log[4096] = {0};
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, NULL);
        log[4095] = 0;
        fprintf(stderr, "OpenCL compilation failed: %.4095s\n", log);
        goto done;
    }
    kernel = clCreateKernel(program, "hyperl_kernel", &error);
    if (error != CL_SUCCESS)
        goto done;
    for (long i = 0; i < inputs; i++) {
        char name[32];
        snprintf(name, sizeof(name), "input_%ld.bin", i);
        if (!filename(file, sizeof(file), argv[5], name))
            goto done;
        float *data = read_exact(file, bytes);
        if (!data)
            goto done;
        for (long j = 0; j < length; j++)
            if (!isfinite(data[j])) {
                free(data);
                goto done;
            }
        buffers[i] =
            clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, data, &error);
        free(data);
        if (error != CL_SUCCESS)
            goto done;
        if (clSetKernelArg(kernel, (cl_uint)i, sizeof(cl_mem), &buffers[i]) != CL_SUCCESS)
            goto done;
    }
    buffers[inputs] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &error);
    if (error != CL_SUCCESS)
        goto done;
    cl_uint n = (cl_uint)length;
    if (clSetKernelArg(kernel, (cl_uint)inputs, sizeof(cl_mem), &buffers[inputs]) != CL_SUCCESS ||
        clSetKernelArg(kernel, (cl_uint)inputs + 1, sizeof(n), &n) != CL_SUCCESS)
        goto done;
    size_t global = (size_t)length;
    if (clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL) != CL_SUCCESS)
        goto done;
    result = malloc(bytes);
    if (!result)
        goto done;
    if (clEnqueueReadBuffer(queue, buffers[inputs], CL_TRUE, 0, bytes, result, 0, NULL, NULL) !=
        CL_SUCCESS)
        goto done;
    for (long i = 0; i < length; i++)
        if (!isfinite(result[i])) {
            fputs("Nonfinite result/intermediate rejected\n", stderr);
            goto done;
        }
    if (!filename(file, sizeof(file), argv[5], "result.bin"))
        goto done;
    f = fopen(file, "wb");
    if (!f)
        goto done;
    int written = fwrite(result, 1, bytes, f) == bytes;
    int closed = fclose(f) == 0;
    if (!written || !closed)
        goto done;
    status = 0;
done:
    for (size_t i = 0; i < HL_MAX_INPUTS + 1; i++)
        if (buffers[i])
            clReleaseMemObject(buffers[i]);
    if (kernel)
        clReleaseKernel(kernel);
    if (program)
        clReleaseProgram(program);
    if (queue)
        clReleaseCommandQueue(queue);
    if (context)
        clReleaseContext(context);
    free(source);
    free(result);
    return status;
}
