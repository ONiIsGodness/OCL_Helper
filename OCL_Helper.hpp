#ifndef OpenCL_Healper_hpp
#define OpenCL_Healper_hpp

#ifdef __APPLE__
	#include <OpenCL/opencl.h>
#else
	#include <CL/opencl.h>
#endif

#include <string>
#include <vector>
#include <iostream>

#define OCL_GLOBAL static
#define ARR_SIZE    10

#ifdef VS_PROJECT

	#pragma warning(disable : 4996 )

#ifdef OCL_HELPER_EXPORT

	#define OCL_CLASS		__declspec(dllexport)
	#define OCL_FN			__declspec(dllexport) 

#else

	#define OCL_CLASS		__declspec(dllimport)
	#define OCL_FN			__declspec(dllimport) 
	#pragma comment(lib, "OCL_Helper.lib")

#endif // OCL_HELPER_EXPORT

#else

	#define OCL_CLASS
	#define OCL_FN

#endif // VS_PROJECT

using namespace std;



enum OCL_Error{
    OCL_ERR_OK = 0,
    OCL_ERR_NO_MEMORY,
    OCL_ERR_INVAILD_FILE,
    OCL_ERR_INVAILD_PARAM,
    OCL_ERR_BUILD_PROGRAM,
    OCL_ERR_INIT,
    OCL_ERR_OBTAIN_KERNEL,
};


class OCL_CLASS OCL_Helper {
public:
	OCL_Helper(int platform = 0, int devices = 0);
	~OCL_Helper();

public:
	/*  Push kernel asigned by 'cl_file' and 'program' to OpenCL.

		1.  * The result of computational task will be returned by 'result'
			whose size is asigned by 'result_size' and value comes from the
			'device_mem'

		2.  * Ensure the value of 'param_index' is 0 for some recursion reason.

		3.  * The argument value of kernel function are given by last 2*n
			paramters and always in pair. */
	template<typename... Args>
	OCL_Error
		PushTask(const string& cl_file, const string& program, size_t result_size, void* result, cl_mem device_mem, int param_index, size_t param_size, const void* value, const Args&... args)
	{
		cl_kernel kernel = NULL;
		if (OCL_ERR_OK != ObtainKernel(cl_file, program, kernel))
			return OCL_ERR_OBTAIN_KERNEL;

		if (SetKernelArg(kernel, param_index, param_size, value) != OCL_ERR_OK)
			return OCL_ERR_INVAILD_PARAM;

		return PushTask(cl_file, program, result_size, result, device_mem, param_index + 1, args...);
	}

public:
	/* Push kernel to OpenCL with no paramters. */
	OCL_Error PushTask(const string& cl_file, const string& program, size_t result_size, void* result, cl_mem device_mem, int param_index);

	OCL_Error CreateMemObject(cl_mem memObject[1], void* host_mem, size_t size);

	OCL_Error ReleaseMemObject(cl_mem memObject);

	OCL_FN friend ostream& operator<<(ostream& os, const OCL_Helper& ocl);

    void test();

public:
	OCL_GLOBAL OCL_Error Print();

private:
    /*  1. Compile program assigned by 'cl_file'
        2. Return the index of 'cl_program' object in 'vCompiledPrograms'
        3. If the target file(*.cl) has been compiled, return the index directly.   */
    OCL_Error CompileProgram( const string& cl_file, int& programIndex );
    
    /*  Lookup for the 'cl_kernel' through given file name and function name*/
    OCL_Error LookupKernel(const string& cl_file, const string& program, cl_kernel& kernel);
    
    /* Select Platform & Device */
    OCL_Error Select(int plt, int dvc);
    
    OCL_Error ObtainKernel(const string& cl_file, const string& program, cl_kernel& kernel);
    
    OCL_Error SetKernelArg(cl_kernel kernel, cl_uint param_index, size_t param_size, const void* param);
    
    OCL_Error Release();
    
private:
    OCL_GLOBAL OCL_Error    Initialize();
    
    OCL_GLOBAL OCL_Error    ReadFileContent(const string& fileName, char*& content, size_t& size);
    
    OCL_GLOBAL void         _stdcall PrintBuildLog(cl_program program,  void* user);
    
private:
    enum OCL_CONSTANCE{
       MAX_DEVICES_PER_PLATFORM = 5,
	   MAX_OCL_PLATFORMS		= 5,
       MSG_BUFFER_SIZE          = 1024,
    };
       
    struct OCL_Platform{
       cl_platform_id       Platform;
       cl_uint              numDevices;
       cl_device_id         Devices[MAX_DEVICES_PER_PLATFORM];
    };
    
    struct OCL_Kernel{
        cl_kernel           Kernel;
        
        string              clFileName;
        string              fnName;
        size_t              programIndex;
    };
    
    struct OCL_Program{
        cl_program          Program;
        string              clFileName;
    };

    vector<OCL_Kernel>      vKernels;
    
    vector<OCL_Program>     vCompiledPrograms;
    
    cl_platform_id          platform;
    cl_device_id            device;
    cl_context              context;
    cl_command_queue        queue;
    
private:
    OCL_GLOBAL vector<OCL_Platform>
                            vPlatforms;
    
    OCL_GLOBAL OCL_Error    eStatus;
};





#endif /* OpenCL_Healper_hpp */
