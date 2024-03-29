class FCPlugin: public IPluginExt
{
public:
    FCPlugin(const Weights *weights, int nbWeights, int nbOutputChannels): mNbOutputChannels(nbOutputChannels)
    {
        assert(nbWeights == 0);//判断权重类型是否有两个
        mNbInputChannels=nbOutputChannels;
        cout<<"1"<<endl;
    }

    // create the plugin at runtime from a byte stream
    //依据字节流创建接口
    FCPlugin(const void* data, size_t length)
    {
        cout<<"2"<<endl;
        const char* d = static_cast<const char*>(data), *a = d;//data转为字符串操作
        read(d, mNbInputChannels);//从d中读取输入元素个数
        read(d, mNbOutputChannels);//从d中读取输出元素个数
        read(d, mDataType);//从d中读取数据类型
        assert(d == a + length);
    }

    ~FCPlugin()
    {
        cout<<"3"<<endl;
    }

    int getNbOutputs() const override
    {
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        cout<<"4"<<endl;
        assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
        assert(mNbInputChannels == inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2]);
        return Dims3(mNbOutputChannels, 1, 1);
    }

    bool supportsFormat(DataType type, PluginFormat format) const override { return (type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW; }
    
    

    void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override
    {
        cout<<"5"<<endl;
        assert((type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW);
        mDataType = type;
    }

    int initialize() override//把偏置和网络模型权重copy到GPU
    {
        cout<<"6"<<endl;
        return 0;
    }

    virtual void terminate() override
    {
        cout<<"7"<<endl;
    }

    virtual size_t getWorkspaceSize(int maxBatchSize) const override
    {
        cout<<"7+"<<endl;
        return 0;
    }

    virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override
    {
        cout<<"8"<<endl;
        float* h_C = (float*)malloc(sizeof(float)*batchSize*mNbOutputChannels);
        CHECK(cudaMemcpy(h_C,inputs[0],sizeof(float)*batchSize*mNbOutputChannels,cudaMemcpyDeviceToHost));
            
        for(int i=0;i<batchSize*mNbOutputChannels;i++)
        {
                if(h_C[i]<0)
                    h_C[i]=0;
        }
        CHECK(cudaMemcpy(outputs[0],h_C,sizeof(float)*batchSize*mNbOutputChannels,cudaMemcpyHostToDevice));     
        return 0;
    }

    virtual size_t getSerializationSize() override
    {
        cout<<"9"<<endl;
        return sizeof(mNbInputChannels) + sizeof(mNbOutputChannels) + sizeof(mDataType);
    }

    virtual void serialize(void* buffer) override
    {
        cout<<"10"<<endl;
        char* d = static_cast<char*>(buffer), *a = d;
        write(d, mNbInputChannels);
        write(d, mNbOutputChannels);
        write(d, mDataType);
        assert(d == a + getSerializationSize());
    }

private:
    size_t type2size(DataType type) { return type == DataType::kFLOAT ? sizeof(float) : sizeof(__half); }

    template<typename T> void write(char*& buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template<typename T> void read(const char*& buffer, T& val)
    {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }

    int mNbOutputChannels, mNbInputChannels;

    DataType mDataType{DataType::kFLOAT};
};

// integration for serialization
class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactoryExt
{
public:
    // caffe parser plugin implementation
    bool isPlugin(const char* name) override//判断该层是不是接口层
    {
        //cout<<"<-1->"<<endl;
        return isPluginExt(name);
    }

    bool isPluginExt(const char* name) override
    {
        //cout<<"relu1"<<endl;
        return !strcmp(name, "relu1");
       
    }

    virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override
    //根据权重和输出单元的个数生成网络layer接口
    {
        // there's no way to pass parameters through from the model definition, so we have to define it here explicitly
        //没有办法从模型定义中传递参数因此这里需要明确定义
        static const int NB_OUTPUT_CHANNELS = 500;//设置输出层通道的个数
        assert(isPlugin(layerName) && nbWeights == 0);
        assert(mPlugin.get() == nullptr);
        mPlugin = std::unique_ptr<FCPlugin>(new FCPlugin(weights, nbWeights, NB_OUTPUT_CHANNELS));//根据权重和输出单元的个数生成网络layer接口
        return mPlugin.get();
    }

    // deserialization plugin implementation
    //依据反序列化数据生成接口
    IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
    {
        assert(isPlugin(layerName));
        assert(mPlugin.get() == nullptr);
        mPlugin = std::unique_ptr<FCPlugin>(new FCPlugin(serialData, serialLength));
        return mPlugin.get();
    }

    // User application destroys plugin when it is safe to do so.
    // Should be done after consumers of plugin (like ICudaEngine) are destroyed.
    void destroyPlugin()
    {
        mPlugin.reset();
    }

    std::unique_ptr<FCPlugin> mPlugin{ nullptr };
};
执行推断前的执行顺序为

1
4
5
7+
7+
6
9
10
9
7
3
执行推断后的顺序为：
2
6
8
7
3

