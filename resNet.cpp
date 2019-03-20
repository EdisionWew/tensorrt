#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <sys/stat.h>
#include <vector>
#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "common.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
static Logger gLogger;

static const int INPUT_H=224;
static const int INPUT_W=224;
static const int INPUT_C=3;
static const int INPUT_N=1;
static const int OUTPUT_SIZE=1000;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
const std::string dir="/root/wangerwei/resnet/";

void ResNetToModel(const std::string &protofile, //caffe网络配置文件
                   const std::string &modelfile, //caffe训练好的模型文件
                   const std::vector<std::string>& outputs, //网络输出
                   unsigned int maxBatchSize,    //最大batchSize
                   IHostMemory*& trtModelStream)  //TRT模型的输出buffer
{
   IBuilder * builder=createInferBuilder(gLogger); //创建builder
   INetworkDefinition* network=builder->createNetwork();//创建网络
   std::string pf=dir+protofile;
   std::string mf=dir+modelfile;//获取配置文件以及网络模型
   ICaffeParser* parser=createCaffeParser();//创建解析器
   const IBlobNameToTensor* BlobNameToTensor=parser->parse(pf.c_str(),mf.c_str(),*network,DataType::kFLOAT);//解析网络并配置
    
   for(auto& s: outputs)
   {
       network->markOutput(*BlobNameToTensor->find(s.c_str()));
   }//指明网络输出对象
    
    
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 20);
    
    
    ICudaEngine* engine=builder->buildCudaEngine(*network);//创建engine
    
    network->destroy();
    parser->destroy();
    
    trtModelStream=engine->serialize();//引擎序列化
    engine->destroy();
    builder->destroy();
    return;
}


/********************************推理*********************************/
void doInference(IExecutionContext& context,float* input,float* output,int batchSize)
{
    const ICudaEngine& engine=context.getEngine();//从上下文获取引擎
    void* buffers[2];
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);//为input数据设置索引
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));//在gpu开辟内存空间
    
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));//赋值输出数据到GPU 大小为 batchSize
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));//从GPU提取数据到cpu
    cudaStreamSynchronize(stream);
    
    
    
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));//释放对应GPU空间
    
}
    
int main(int argc, char** argv)
{
    IHostMemory * ModelStream(nullptr);
    ResNetToModel("resnet50.prototxt","resnet50.caffemodel",std::vector<std::string>{OUTPUT_BLOB_NAME},1,ModelStream);
        
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(ModelStream->data(), ModelStream->size(), nullptr);
    assert(engine != nullptr);
    ModelStream->destroy();
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    
    float* prob=new float[OUTPUT_SIZE];
    float* data=new float[INPUT_C * INPUT_H * INPUT_W];
    int batchSize=1;
    for(int i=0;i<INPUT_C*INPUT_H*INPUT_W;i++)
        data[i]=(float)(rand()%255)*1.0;
    
    doInference(*context,data,prob,batchSize);
    
        
    std::cout << "\nOutput:\n\n";
    float val=0.0;
    int maxIdex=0;
    for (unsigned int i = 0; i < 1000; i++)
    {
           if(val<prob[i])
           {
               val=prob[i];
               maxIdex=i;
           }
    }
    
    std::cout<<maxIdex<<"  "<<val<<std::endl;
    
    delete [] prob;
    delete [] data;
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}
