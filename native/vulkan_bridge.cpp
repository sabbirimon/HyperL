// SPDX-License-Identifier: Apache-2.0
// Original bounded Vulkan qualification host. Not a model engine or app adapter.
#include <vulkan/vulkan.h>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

static void check(VkResult result, const char* stage) {
    if (result != VK_SUCCESS) throw std::runtime_error(std::string(stage) + ": VkResult " + std::to_string(result));
}
static uint32_t integer(const char* text, uint32_t low, uint32_t high) {
    std::string s(text); if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos) throw std::runtime_error("Invalid integer");
    unsigned long long n = std::stoull(s);
    if (n < low || n > high) throw std::runtime_error("Integer outside bounds");
    return static_cast<uint32_t>(n);
}
static std::string quoted(const char* value) {
    std::string out = "\"";
    for (const unsigned char c : std::string(value)) {
        if (c == '\"' || c == '\\') { out += '\\'; out += static_cast<char>(c); }
        else if (c < 32) { char escaped[7]; std::snprintf(escaped, sizeof(escaped), "\\u%04x", c); out += escaped; }
        else out += static_cast<char>(c);
    }
    return out + "\"";
}
static bool hardware(const VkPhysicalDeviceProperties& p) {
    std::string name(p.deviceName); std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c){return static_cast<char>(std::tolower(c));});
    return (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU || p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        && name.find("swiftshader") == std::string::npos && name.find("llvmpipe") == std::string::npos
        && name.find("lavapipe") == std::string::npos;
}
static uint32_t computeFamily(VkPhysicalDevice physical) {
    uint32_t count = 0; vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
    if (!count || count > 128) throw std::runtime_error("Queue family count outside bounds");
    std::vector<VkQueueFamilyProperties> families(count); vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
    for (uint32_t i=0; i<count; ++i) if (families[i].queueCount && (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) return i;
    throw std::runtime_error("No compute queue");
}
struct Buffer { VkBuffer handle = VK_NULL_HANDLE; VkDeviceMemory memory = VK_NULL_HANDLE; void* mapped = nullptr; };
struct Context {
    VkInstance instance = VK_NULL_HANDLE; VkDevice device = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptors = VK_NULL_HANDLE; VkPipelineLayout layout = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE; VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE; VkCommandPool commands = VK_NULL_HANDLE; VkFence fence = VK_NULL_HANDLE;
    std::vector<Buffer> buffers;
    ~Context() {
        if (device) {
            // Submitted work is completed before reaching this destructor. A timed-out
            // request exits the process instead of hanging in vkDeviceWaitIdle.
            if (fence) vkDestroyFence(device, fence, nullptr);
            if (commands) vkDestroyCommandPool(device, commands, nullptr);
            if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
            if (shader) vkDestroyShaderModule(device, shader, nullptr);
            if (pool) vkDestroyDescriptorPool(device, pool, nullptr);
            if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
            if (descriptors) vkDestroyDescriptorSetLayout(device, descriptors, nullptr);
            for (auto& b : buffers) { if (b.mapped) vkUnmapMemory(device,b.memory); if (b.handle) vkDestroyBuffer(device,b.handle,nullptr); if (b.memory) vkFreeMemory(device,b.memory,nullptr); }
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }
};
template<typename T> static std::vector<T> read(const std::string& path, size_t count) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(count*sizeof(T))) throw std::runtime_error("File size mismatch: " + path);
    std::vector<T> data(count); file.seekg(0); file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count*sizeof(T)));
    if (!file) throw std::runtime_error("Read failed: " + path); return data;
}
static std::vector<uint32_t> readShader(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Shader unavailable");
    const auto bytes = file.tellg();
    if (bytes < 20 || bytes > 262144 || bytes%4 != 0) throw std::runtime_error("Shader size outside bounds");
    auto code = read<uint32_t>(path, static_cast<size_t>(bytes)/4);
    if (code[0] != 0x07230203) throw std::runtime_error("Invalid SPIR-V magic"); return code;
}
static void execute(Context& c, VkPhysicalDevice physical, const VkPhysicalDeviceProperties& props,
                    uint32_t n, uint32_t inputCount, const std::string& directory) {
    const uint16_t endian = 1;
    if (*reinterpret_cast<const uint8_t*>(&endian) != 1 || sizeof(float) != 4 || !std::numeric_limits<float>::is_iec559) throw std::runtime_error("Requires little-endian IEEE f32");
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(n)*4;
    const auto& limits = props.limits;
    if (!hardware(props)) throw std::runtime_error("Selected device is not a hardware GPU; no fallback");
    if (limits.maxComputeWorkGroupInvocations < 64 || limits.maxComputeWorkGroupSize[0] < 64 ||
        limits.maxComputeWorkGroupCount[0] < (n+63)/64 || limits.maxPushConstantsSize < 4 ||
        limits.maxPerStageDescriptorStorageBuffers < inputCount+1 || limits.maxDescriptorSetStorageBuffers < inputCount+1 ||
        limits.maxStorageBufferRange < bytes) throw std::runtime_error("Device limits insufficient");
    auto code = readShader(directory + "/kernel.spv");
    std::vector<std::vector<float>> inputs;
    for (uint32_t i=0; i<inputCount; ++i) {
        inputs.push_back(read<float>(directory + "/input_" + std::to_string(i) + ".bin",n));
        for (float v : inputs.back()) if (!std::isfinite(v)) throw std::runtime_error("Nonfinite input");
    }
    uint32_t family = computeFamily(physical); float priority = 1;
    VkDeviceQueueCreateInfo queueInfo{}; queueInfo.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO; queueInfo.queueFamilyIndex=family; queueInfo.queueCount=1; queueInfo.pQueuePriorities=&priority;
    VkDeviceCreateInfo deviceInfo{}; deviceInfo.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO; deviceInfo.queueCreateInfoCount=1; deviceInfo.pQueueCreateInfos=&queueInfo;
    check(vkCreateDevice(physical,&deviceInfo,nullptr,&c.device),"create device");
    VkQueue queue{}; vkGetDeviceQueue(c.device,family,0,&queue);
    VkPhysicalDeviceMemoryProperties memoryProps{}; vkGetPhysicalDeviceMemoryProperties(physical,&memoryProps);
    VkDeviceSize allocated=0; c.buffers.resize(inputCount+1);
    for (uint32_t i=0; i<=inputCount; ++i) {
        auto& b=c.buffers[i]; VkBufferCreateInfo info{}; info.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; info.size=bytes; info.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; info.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(c.device,&info,nullptr,&b.handle),"create buffer");
        VkMemoryRequirements req{}; vkGetBufferMemoryRequirements(c.device,b.handle,&req);
        if (req.size > 32*1024*1024 || allocated > 32*1024*1024-req.size) throw std::runtime_error("32 MiB device allocation budget exceeded"); allocated+=req.size;
        uint32_t type=UINT32_MAX;
        for (uint32_t t=0; t<memoryProps.memoryTypeCount; ++t) {
            const auto flags=memoryProps.memoryTypes[t].propertyFlags;
            if ((req.memoryTypeBits & (1u<<t)) && (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {type=t;break;}
        }
        if (type==UINT32_MAX) throw std::runtime_error("Host-visible coherent storage memory unavailable");
        VkMemoryAllocateInfo alloc{}; alloc.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; alloc.allocationSize=req.size; alloc.memoryTypeIndex=type;
        check(vkAllocateMemory(c.device,&alloc,nullptr,&b.memory),"allocate memory"); check(vkBindBufferMemory(c.device,b.handle,b.memory,0),"bind memory");
        check(vkMapMemory(c.device,b.memory,0,bytes,0,&b.mapped),"map memory");
        if (i<inputCount) std::memcpy(b.mapped,inputs[i].data(),static_cast<size_t>(bytes));
        else std::fill_n(static_cast<float*>(b.mapped),n,std::numeric_limits<float>::quiet_NaN());
    }
    std::vector<VkDescriptorSetLayoutBinding> bindings(inputCount+1);
    for (uint32_t i=0; i<=inputCount; ++i) { bindings[i].binding=i; bindings[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bindings[i].descriptorCount=1; bindings[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo descriptorInfo{}; descriptorInfo.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO; descriptorInfo.bindingCount=inputCount+1; descriptorInfo.pBindings=bindings.data();
    check(vkCreateDescriptorSetLayout(c.device,&descriptorInfo,nullptr,&c.descriptors),"create descriptor layout");
    VkPushConstantRange push{}; push.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT; push.size=4;
    VkPipelineLayoutCreateInfo layoutInfo{}; layoutInfo.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO; layoutInfo.setLayoutCount=1; layoutInfo.pSetLayouts=&c.descriptors; layoutInfo.pushConstantRangeCount=1; layoutInfo.pPushConstantRanges=&push;
    check(vkCreatePipelineLayout(c.device,&layoutInfo,nullptr,&c.layout),"create pipeline layout");
    VkShaderModuleCreateInfo shaderInfo{}; shaderInfo.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO; shaderInfo.codeSize=code.size()*4; shaderInfo.pCode=code.data();
    check(vkCreateShaderModule(c.device,&shaderInfo,nullptr,&c.shader),"create shader");
    VkComputePipelineCreateInfo pipelineInfo{}; pipelineInfo.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO; pipelineInfo.layout=c.layout; pipelineInfo.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; pipelineInfo.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT; pipelineInfo.stage.module=c.shader; pipelineInfo.stage.pName="main";
    const auto compileStart=std::chrono::steady_clock::now();
    check(vkCreateComputePipelines(c.device,VK_NULL_HANDLE,1,&pipelineInfo,nullptr,&c.pipeline),"create pipeline");
    const double compileMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-compileStart).count();
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,inputCount+1}; VkDescriptorPoolCreateInfo poolInfo{}; poolInfo.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO; poolInfo.maxSets=1; poolInfo.poolSizeCount=1; poolInfo.pPoolSizes=&size;
    check(vkCreateDescriptorPool(c.device,&poolInfo,nullptr,&c.pool),"create descriptor pool");
    VkDescriptorSet set{}; VkDescriptorSetAllocateInfo setInfo{}; setInfo.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO; setInfo.descriptorPool=c.pool; setInfo.descriptorSetCount=1; setInfo.pSetLayouts=&c.descriptors;
    check(vkAllocateDescriptorSets(c.device,&setInfo,&set),"allocate descriptors");
    std::vector<VkDescriptorBufferInfo> infos(inputCount+1); std::vector<VkWriteDescriptorSet> writes(inputCount+1);
    for (uint32_t i=0;i<=inputCount;++i) { infos[i]={c.buffers[i].handle,0,bytes}; writes[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[i].dstSet=set; writes[i].dstBinding=i; writes[i].descriptorCount=1; writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[i].pBufferInfo=&infos[i]; }
    vkUpdateDescriptorSets(c.device,inputCount+1,writes.data(),0,nullptr);
    VkCommandPoolCreateInfo commandPool{}; commandPool.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; commandPool.queueFamilyIndex=family;
    check(vkCreateCommandPool(c.device,&commandPool,nullptr,&c.commands),"create command pool");
    VkCommandBuffer command{}; VkCommandBufferAllocateInfo commandInfo{}; commandInfo.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO; commandInfo.commandPool=c.commands; commandInfo.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; commandInfo.commandBufferCount=1;
    check(vkAllocateCommandBuffers(c.device,&commandInfo,&command),"allocate command");
    VkCommandBufferBeginInfo begin{}; begin.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO; begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(command,&begin),"begin command");
    vkCmdBindPipeline(command,VK_PIPELINE_BIND_POINT_COMPUTE,c.pipeline); vkCmdBindDescriptorSets(command,VK_PIPELINE_BIND_POINT_COMPUTE,c.layout,0,1,&set,0,nullptr); vkCmdPushConstants(command,c.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,4,&n); vkCmdDispatch(command,(n+63)/64,1,1);
    VkMemoryBarrier barrier{}; barrier.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER; barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&barrier,0,nullptr,0,nullptr);
    check(vkEndCommandBuffer(command),"end command");
    VkFenceCreateInfo fenceInfo{}; fenceInfo.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO; check(vkCreateFence(c.device,&fenceInfo,nullptr,&c.fence),"create fence");
    VkSubmitInfo submit{}; submit.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO; submit.commandBufferCount=1; submit.pCommandBuffers=&command;
    const auto start=std::chrono::steady_clock::now(); check(vkQueueSubmit(queue,1,&submit,c.fence),"submit");
    const auto wait=vkWaitForFences(c.device,1,&c.fence,VK_TRUE,2000000000ULL);
    if (wait!=VK_SUCCESS) { std::cerr<<"GPU completion not confirmed: "<<wait<<std::endl; std::_Exit(3); }
    const double submitMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    const float* result=static_cast<float*>(c.buffers.back().mapped);
    for (uint32_t i=0;i<n;++i) if (!std::isfinite(result[i])) throw std::runtime_error("Nonfinite output; overflow or incomplete output rejected");
    const std::string resultPath=directory+"/result.bin";
    const int output=::open(resultPath.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);
    if(output<0) throw std::runtime_error("Cannot create fresh result");
    size_t offset=0; bool failed=false;
    while(offset<bytes) { const auto written=::write(output,reinterpret_cast<const char*>(result)+offset,static_cast<size_t>(bytes)-offset); if(written<=0){failed=true;break;} offset+=static_cast<size_t>(written); }
    if(::close(output)!=0) failed=true;
    if(failed) {::unlink(resultPath.c_str()); throw std::runtime_error("Result write failed");}
    std::cout<<"{\"format\":\"hyperl-vulkan-result/1\",\"backend\":\"VULKAN_GPU\",\"device\":"<<quoted(props.deviceName)<<",\"gpuCompletionConfirmed\":true,\"elements\":"<<n<<",\"allocatedBytes\":"<<allocated<<",\"storageMode\":\"host-visible-coherent\",\"compileMs\":"<<compileMs<<",\"submitAndWaitMs\":"<<submitMs<<"}\n";
}
int main(int argc,char** argv) {
    try {
        if (!((argc==2 && std::string(argv[1])=="--probe") || (argc==6 && std::string(argv[1])=="--run"))) throw std::runtime_error("Usage: --probe | --run DEVICE_INDEX N INPUT_COUNT DIRECTORY");
        Context c; VkApplicationInfo app{}; app.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO; app.pApplicationName="HyperL qualification"; app.apiVersion=VK_API_VERSION_1_0;
        VkInstanceCreateInfo info{}; info.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; info.pApplicationInfo=&app; check(vkCreateInstance(&info,nullptr,&c.instance),"create instance");
        uint32_t count=0; check(vkEnumeratePhysicalDevices(c.instance,&count,nullptr),"enumerate count");
        if (count>32) throw std::runtime_error("Device count limit"); std::vector<VkPhysicalDevice> devices(count);
        if(count) check(vkEnumeratePhysicalDevices(c.instance,&count,devices.data()),"enumerate devices");
        if(argc==2) {
            std::cout<<"{\"format\":\"hyperl-vulkan-probe/1\",\"inferenceQualified\":false,\"devices\":[";
            for(uint32_t i=0;i<count;++i) { VkPhysicalDeviceProperties p{}; vkGetPhysicalDeviceProperties(devices[i],&p); bool compute=true; try {computeFamily(devices[i]);}catch(const std::exception&){compute=false;}
                if(i) std::cout<<",";
                std::cout<<"{\"index\":"<<i<<",\"name\":"<<quoted(p.deviceName)<<",\"hardwareGpu\":"<<(hardware(p)?"true":"false")<<",\"computeQueue\":"<<(compute?"true":"false")<<",\"vendorId\":"<<p.vendorID<<",\"deviceId\":"<<p.deviceID<<",\"deviceType\":"<<p.deviceType<<",\"apiVersion\":"<<p.apiVersion<<",\"driverVersion\":"<<p.driverVersion<<",\"maxStorageBufferRange\":"<<p.limits.maxStorageBufferRange<<"}";
            }
            std::cout<<"]}\n";
        } else {
            uint32_t index=integer(argv[2],0,31),n=integer(argv[3],1,262144),inputs=integer(argv[4],1,8);
            if(index>=count) throw std::runtime_error("Selected GPU unavailable"); VkPhysicalDeviceProperties p{}; vkGetPhysicalDeviceProperties(devices[index],&p); execute(c,devices[index],p,n,inputs,argv[5]);
        }
        return 0;
    } catch(const std::exception& error) { std::cerr<<"HyperL Vulkan failed: "<<error.what()<<"\n"; return 2; }
}
