/* Original HyperL Metal host bridge. Apache-2.0. Explicit macOS GPU; no CPU fallback. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <cmath>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

static double elapsed(std::chrono::steady_clock::time_point start){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();}
static long number(const char *s,long max){char *end;errno=0;long n=strtol(s,&end,10);return errno||!*s||*end||n<0||n>max?-1:n;}
static int fail(const char *message,int status=4){fprintf(stderr,"%s\n",message);return status;}
static int error(const char *stage,NSError *e){fprintf(stderr,"%s: %.2048s\n",stage,e.localizedDescription.UTF8String ?: "Metal operation unavailable");return 4;}
static bool readFile(NSString *path,std::vector<unsigned char>& bytes,size_t max,size_t exact=0){
    int fd=open(path.fileSystemRepresentation,O_RDONLY|O_NOFOLLOW);if(fd<0)return false;
    struct stat st;bool valid=fstat(fd,&st)==0 && S_ISREG(st.st_mode) && st.st_size>0 && (uint64_t)st.st_size<=max && (!exact || (uint64_t)st.st_size==exact);
    if(valid){bytes.resize((size_t)st.st_size);size_t pos=0;while(pos<bytes.size()){ssize_t n=read(fd,bytes.data()+pos,bytes.size()-pos);if(n<=0){valid=false;break;}pos+=(size_t)n;}}
    if(valid){unsigned char extra;valid=read(fd,&extra,1)==0;}close(fd);return valid;
}
static bool writeNew(NSString *path,const void *data,size_t length){
    int fd=open(path.fileSystemRepresentation,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);if(fd<0)return false;
    size_t pos=0;bool ok=true;while(pos<length){ssize_t n=write(fd,(const char*)data+pos,length-pos);if(n<=0){ok=false;break;}pos+=(size_t)n;}
    if(close(fd))ok=false;return ok;
}
static NSArray<id<MTLDevice>> *devices(){NSArray<id<MTLDevice>> *all=MTLCopyAllDevices();return all.count<=128 ? all : @[];}
static NSDictionary *describe(id<MTLDevice> d,NSUInteger index){return @{@"index":@(index),@"name":d.name,@"lowPower":@(d.lowPower),@"hasUnifiedMemory":@(d.hasUnifiedMemory),@"maxBufferLength":@(d.maxBufferLength),@"recommendedMaxWorkingSetSize":@(d.recommendedMaxWorkingSetSize)};}

int main(int argc,char **argv){@autoreleasepool{
    if(argc==2 && !strcmp(argv[1],"--probe")){
        NSMutableArray *list=[NSMutableArray array];NSArray *all=devices();
        for(NSUInteger i=0;i<all.count;i++)[list addObject:describe(all[i],i)];
        NSDictionary *probe=@{@"format":@"hyperl-metal-probe/1",@"backend":@"Metal GPU",@"available":@(all.count>0),@"inferenceQualified":@NO,@"devices":list};
        NSData *json=[NSJSONSerialization dataWithJSONObject:probe options:0 error:nil];fwrite(json.bytes,1,json.length,stdout);putchar('\n');return 0;
    }
    if(argc!=6 || strcmp(argv[1],"--run"))return fail("Use --probe or --run EXACT_DEVICE_NAME N INPUT_COUNT PRIVATE_DIRECTORY",2);
    long length=number(argv[3],262144),inputs=number(argv[4],8);
    NSString *name=[NSString stringWithUTF8String:argv[2]],*dir=[NSString stringWithUTF8String:argv[5]];
    uint32_t endian=1;
    if(!name.length || name.length>256 || !dir || length<1 || inputs<1 || length*inputs>1048576 || *(unsigned char*)&endian!=1)return fail("Invalid bounded arguments or unsupported host byte order",2);
    id<MTLDevice> selected=nil;
    for(id<MTLDevice> d in devices())if([d.name isEqualToString:name]){if(selected)return fail("Device name is ambiguous; no device selected",3);selected=d;}
    if(!selected)return fail("Exact Metal GPU unavailable in this process; no CPU or other-GPU fallback",3);
    size_t size=(size_t)length*sizeof(float);
    if(size>selected.maxBufferLength)return fail("Device buffer limit",3);
    auto started=std::chrono::steady_clock::now();std::vector<unsigned char> source;
    if(!readFile([dir stringByAppendingPathComponent:@"kernel.metal"],source,65536))return fail("Bounded regular kernel source required");
    NSString *text=[[NSString alloc] initWithBytes:source.data() length:source.size() encoding:NSUTF8StringEncoding];if(!text)return fail("UTF-8 Metal source required");
    MTLCompileOptions *options=[MTLCompileOptions new];options.fastMathEnabled=NO;options.languageVersion=MTLLanguageVersion2_0;
    NSError *e=nil;id<MTLLibrary> library=[selected newLibraryWithSource:text options:options error:&e];if(!library)return error("Metal compilation failed",e);
    id<MTLFunction> function=[library newFunctionWithName:@"hyperl_kernel"];if(!function)return fail("hyperl_kernel entry unavailable");
    id<MTLComputePipelineState> pipeline=[selected newComputePipelineStateWithFunction:function error:&e];if(!pipeline)return error("Metal pipeline failed",e);
    double compileMs=elapsed(started);
    // Intel/AMD discrete GPUs use managed copies and explicit CPU/GPU synchronization.
    BOOL managed=!selected.hasUnifiedMemory;
    MTLResourceOptions storage=managed?MTLResourceStorageModeManaged:MTLResourceStorageModeShared;
    NSMutableArray<id<MTLBuffer>> *buffers=[NSMutableArray array];
    for(long i=0;i<inputs;i++){
        std::vector<unsigned char> data;
        if(!readFile([dir stringByAppendingPathComponent:[NSString stringWithFormat:@"input_%ld.bin",i]],data,size,size))return fail("Exact bounded input file required");
        for(long j=0;j<length;j++){float v;memcpy(&v,data.data()+j*sizeof(float),sizeof(v));if(!std::isfinite(v))return fail("Finite f32 input required");}
        id<MTLBuffer> buffer=[selected newBufferWithBytes:data.data() length:size options:storage];if(!buffer)return fail("Input buffer allocation failed");
        if(managed)[buffer didModifyRange:NSMakeRange(0,size)];[buffers addObject:buffer];
    }
    id<MTLBuffer> result=[selected newBufferWithLength:size options:storage];if(!result || !result.contents)return fail("Output buffer allocation failed");
    for(long i=0;i<length;i++)((float*)result.contents)[i]=NAN;
    if(managed)[result didModifyRange:NSMakeRange(0,size)];
    id<MTLCommandQueue> queue=[selected newCommandQueue];id<MTLCommandBuffer> command=[queue commandBuffer];id<MTLComputeCommandEncoder> compute=[command computeCommandEncoder];
    if(!queue || !command || !compute)return fail("Metal command allocation failed");
    [compute setComputePipelineState:pipeline];
    for(NSUInteger i=0;i<buffers.count;i++)[compute setBuffer:buffers[i] offset:0 atIndex:i];
    [compute setBuffer:result offset:0 atIndex:(NSUInteger)inputs];uint32_t n=(uint32_t)length;[compute setBytes:&n length:sizeof(n) atIndex:(NSUInteger)inputs+1];
    NSUInteger group=MIN((NSUInteger)256,pipeline.maxTotalThreadsPerThreadgroup);if(!group)return fail("Threadgroup capability unavailable");
    [compute dispatchThreadgroups:MTLSizeMake(((NSUInteger)length+group-1)/group,1,1) threadsPerThreadgroup:MTLSizeMake(group,1,1)];[compute endEncoding];
    if(managed){id<MTLBlitCommandEncoder> sync=[command blitCommandEncoder];if(!sync)return fail("Managed output synchronization unavailable");[sync synchronizeResource:result];[sync endEncoding];}
    NSString *marker=[dir stringByAppendingPathComponent:@"submitted.marker"];
    if(!writeNew(marker,"pending\n",8))return fail("Submission marker could not be created");
    auto submit=std::chrono::steady_clock::now();[command commit];[command waitUntilCompleted];double waitMs=elapsed(submit);
    if(command.status!=MTLCommandBufferStatusCompleted && command.status!=MTLCommandBufferStatusError)return fail("GPU completion unconfirmed");
    if(unlink(marker.fileSystemRepresentation))return fail("GPU completion marker cleanup failed");
    if(command.status!=MTLCommandBufferStatusCompleted)return error("Metal command failed",command.error);
    for(long i=0;i<length;i++)if(!std::isfinite(((float*)result.contents)[i]))return fail("Nonfinite or unwritten GPU result rejected");
    double gpuMs=(command.GPUEndTime-command.GPUStartTime)*1000.0;
    NSDictionary *report=@{@"format":@"hyperl-metal-result/1",@"device":selected.name,@"storageMode":managed?@"managed":@"shared",@"gpuCompletionConfirmed":@YES,@"compileMs":@(compileMs),@"submitAndWaitMs":@(waitMs),@"gpuMs":(gpuMs>0 && std::isfinite(gpuMs))?@(gpuMs):[NSNull null]};
    NSData *json=[NSJSONSerialization dataWithJSONObject:report options:0 error:&e];if(!json)return error("Metadata serialization failed",e);
    if(!writeNew([dir stringByAppendingPathComponent:@"result.bin"],result.contents,size))return fail("Result publication failed");
    fwrite(json.bytes,1,json.length,stdout);putchar('\n');return 0;
}}
