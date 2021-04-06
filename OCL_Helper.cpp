//
//  OpenCL_Healper.cpp
//  OpenCL
//
//  Created by Legendre on 2020/12/22.
//  Copyright © 2020 Legendre. All rights reserved.
//

#include "OpenCL_Healper.hpp"
#include <stdio.h>
#include <assert.h>
#include <functional>
#include <unistd.h>

#define ARRAY_SIZE 10

vector<OCL_Helper::OCL_Platform> OCL_Helper::vPlatforms;
OCL_Error OCL_Helper::eStatus = OCL_Helper::Initialize();


OCL_Helper::OCL_Helper(int PlatformIndex, int DeviceIndex):
    platform(0),
    device(0),
    context(0),
    queue(0)
{
    if( eStatus != OCL_ERR_OK )
        return;
 
    cl_int err;
    if( OCL_ERR_OK != Select(PlatformIndex, DeviceIndex))
        return;
    
    cl_context_properties context_properties[] = {
        CL_CONTEXT_PLATFORM,
        cl_context_properties(platform),
        0
    };
    context = clCreateContext(context_properties, 1, &device, NULL , NULL, &err );
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if( err != CL_SUCCESS ){
        clReleaseContext(context);
        return;
    }
}

OCL_Helper::~OCL_Helper()
{
    for(auto pro : vCompiledPrograms)
        clReleaseProgram(pro.Program);
    
    for(auto kernel : vKernels)
        clReleaseKernel(kernel.Kernel);
    
    if( 0 != queue ){
        clReleaseCommandQueue(queue);
        queue = 0;
    }
    
    if( 0 != context ){
        clReleaseContext(context);
        context = 0;
    }
    
}

void OCL_Helper::PrintBuildLog(cl_program program,  void* user)
{
    OCL_Helper* pThis = (OCL_Helper*)user;
    size_t size;
        
    clGetProgramBuildInfo( program, pThis->device, CL_PROGRAM_BUILD_LOG , 0, NULL, &size );
    char* buf = new char[size];
    clGetProgramBuildInfo(program, pThis->device, CL_PROGRAM_BUILD_LOG, size, buf, NULL);
    printf("%s\n", buf);
    delete[] buf;
    buf = NULL;
}

OCL_Error OCL_Helper::Print()
{
    if( eStatus != OCL_ERR_OK &&
       (eStatus = Initialize()) != OCL_ERR_OK )
       return OCL_ERR_INIT;
    
    char msg[MSG_BUFFER_SIZE] = { 0 };
    size_t size;
    
    for( auto plt : vPlatforms ){
        clGetPlatformInfo(plt.Platform, CL_PLATFORM_NAME, MSG_BUFFER_SIZE, msg, &size);
        msg[size - 1] = ',';
        msg[size] = ' ';
        clGetPlatformInfo(plt.Platform, CL_PLATFORM_VERSION, MSG_BUFFER_SIZE - size, msg + size + 1, &size);
        printf("%s\n", msg);
        
        for(int i = 0 ; i < plt.numDevices; ++i){
            clGetDeviceInfo(plt.Devices[i], CL_DEVICE_NAME, MSG_BUFFER_SIZE, msg, NULL);
            printf("\t%s;\n", msg);
        }
    }
    return OCL_ERR_OK;
}

OCL_Error OCL_Helper::Initialize()
{
    cl_int err;
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if( err != CL_SUCCESS || num_platforms <= 0 )
        return OCL_ERR_OK;
    
    cl_platform_id* platforms = new cl_platform_id[num_platforms];
    clGetPlatformIDs(num_platforms, platforms, NULL);
    for(int i = 0 ; i < num_platforms; ++i){
        vPlatforms.push_back(OCL_Platform());
        
        vPlatforms[i].Platform = platforms[i];
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, MAX_DEVICES_PER_PLATFORM, vPlatforms[i].Devices, &vPlatforms[i].numDevices);
    }
    delete[] platforms;
    
    
    return OCL_ERR_OK;
}

OCL_Error OCL_Helper::ReadFileContent(const string& fileName, char*& content, size_t& size)
{
    FILE* fd = fopen(fileName.c_str(), "r");
    if (NULL == fd)
        return OCL_ERR_INVAILD_FILE;

    fseek(fd, 0, SEEK_END);
    size = ftell(fd);
    content = new char[size + 1];
    
    rewind(fd);
    fread(content, size, 1, fd);
    content[size] = '\0';
    fclose(fd);

    return OCL_ERR_OK;
}



OCL_Error OCL_Helper::Release()
{
    if( 0 != queue )
        clReleaseCommandQueue(queue);
    if( 0 != context )
        clReleaseContext(context);
    return OCL_ERR_OK;
}


OCL_Error OCL_Helper::Select(int plt, int dvc)
{
    assert( plt >= 0 && dvc >= 0 );
    assert( plt < vPlatforms.size() && dvc < vPlatforms[plt].numDevices );
    
    platform = vPlatforms[plt].Platform;
    device = vPlatforms[plt].Devices[dvc];
    
    return OCL_ERR_OK;
}


OCL_Error OCL_Helper::LookupKernel(const string& cl_file, const string& program, cl_kernel& kernel)
{
    kernel = 0;
    for( const auto& k : vKernels ){
        if( k.clFileName == cl_file && k.fnName == program ){
            kernel = k.Kernel;
            break;
        }
    }
    return OCL_ERR_OK;
}

OCL_Error OCL_Helper::CompileProgram( const string& cl_file, int& programIndex )
{
    programIndex = 0;
    for( ; programIndex < vCompiledPrograms.size(); ++programIndex){
        if( vCompiledPrograms[programIndex].clFileName == cl_file){
            return OCL_ERR_OK;
        }
    }
    
    OCL_Program program;
    char* content = NULL;
    size_t size;
    
    if( OCL_ERR_OK !=  ReadFileContent(cl_file, content, size))
        return OCL_ERR_INVAILD_FILE;
    
    cl_int err;
    program.clFileName = cl_file;
    program.Program = clCreateProgramWithSource(context, 1, (const char**)&content, (const size_t*)&size, &err);
    clBuildProgram(program.Program, 1, &device, "", &OCL_Helper::PrintBuildLog, this);
    
    vCompiledPrograms.push_back(program);
    return OCL_ERR_OK;
}


ostream& operator<< ( ostream& os,  const OCL_Helper& ocl)
{
    char msg[ OCL_Helper::MSG_BUFFER_SIZE] = { 0 };
    
    if( NULL == ocl.device || NULL == ocl.platform ){
        strcat(msg, "Empty OCL_Helper Object\n.");
    }
    else{
        size_t size;
        clGetPlatformInfo(ocl.platform, CL_PLATFORM_NAME, OCL_Helper::MSG_BUFFER_SIZE, msg, &size);
        msg[size - 1] = ',';
        msg[size] = ' ';
        clGetDeviceInfo(ocl.device, CL_DEVICE_NAME, OCL_Helper::MSG_BUFFER_SIZE - size, msg + size + 1, NULL);
    }
    os << msg;
    return os;
}
    
OCL_Error OCL_Helper::ObtainKernel(const string& cl_file, const string& program, cl_kernel& kernel)
{
    kernel = NULL;
    cl_int err;
    LookupKernel(cl_file, program, kernel);

    if( NULL == kernel ){
        int programIndex;
        if( OCL_ERR_OK != CompileProgram(cl_file, programIndex))
            return OCL_ERR_BUILD_PROGRAM;

        vKernels.push_back(OCL_Kernel());
        OCL_Kernel& k = vKernels.back();
        k.clFileName = cl_file;
        k.fnName = program;

        k.programIndex = programIndex;
        kernel = k.Kernel = clCreateKernel(vCompiledPrograms[programIndex].Program, program.c_str(), &err);
    }
    return OCL_ERR_OK;
}


OCL_Error OCL_Helper::PushTask( const string& cl_file, const string& program, size_t result_size, void* result, cl_mem device_mem, int param_index )
{
    cl_kernel kernel = NULL;
    if(OCL_ERR_OK != ObtainKernel(cl_file, program, kernel))
        return OCL_ERR_OBTAIN_KERNEL;
    
    size_t globalWorkSize[1] = { 1024 };
    size_t localWorkSize[1] = { 1 };
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if( (nullptr != result) && (device_mem != nullptr) )
        clEnqueueReadBuffer(queue, device_mem, CL_TRUE, 0, result_size, result, 0, NULL, NULL);
    
    return OCL_ERR_OK;
}
    
OCL_Error OCL_Helper::CreateMemObject(cl_mem memObject[1], void* host_mem, size_t size)
{
    cl_mem_flags flag;
    if( nullptr == host_mem )
        flag = CL_MEM_READ_WRITE;
    else
        flag = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
    
    memObject[0] = clCreateBuffer(context, flag, size, host_mem, NULL);
    if( NULL == memObject[0] )
        return OCL_ERR_NO_MEMORY;
    return OCL_ERR_OK;
}
    
OCL_Error OCL_Helper::ReleaseMemObject( cl_mem memObject )
{
    if( CL_SUCCESS != clReleaseMemObject(memObject))
        return OCL_ERR_INVAILD_PARAM;
    return OCL_ERR_OK;
}
    
    
OCL_Error OCL_Helper::SetKernelArg(cl_kernel kernel, cl_uint index, size_t size, const void* value)
{
    return clSetKernelArg(kernel, index, size, value) == CL_SUCCESS ? OCL_ERR_OK : OCL_ERR_INVAILD_PARAM;
}


void OCL_Helper::test()
{
    float buffer[100] = { 0 };
    float* output = nullptr;
    cl_int err;
    cl_mem bufferA;

    bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(buffer), buffer, &err);
    assert( err == CL_SUCCESS);
    
    output = (cl_float*) clEnqueueMapBuffer(queue, bufferA, CL_TRUE, CL_MAP_WRITE, 0, sizeof(buffer), 0, NULL, NULL, &err);
    assert( err == CL_SUCCESS);

    cout << buffer << "; "<< output << endl;
}
