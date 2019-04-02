class ConvPlugin: public IPluginExt
{
public:
    ConvPlugin(const Weights *weights, int nbWeights,int kernel_size,int stride,int pad,int group,int dilation,int nbOutputChannels):
    kernel_size_(kernel_size),stride_(stride),pad_(pad),group_(group),dilation_(dilation),mNbOutputChannels(nbOutputChannels)
    {
    
        printf("1\n");
        assert(nbWeights == 2);//判断权重类型是否有两个

        mKernelWeights = weights[0];//该层的核权重
        assert(mKernelWeights.type == DataType::kFLOAT || mKernelWeights.type == DataType::kHALF);
        
        
        
        mBiasWeights = weights[1];//该层对应的偏置的权重
        assert(mBiasWeights.count == 0 || mBiasWeights.count == nbOutputChannels);
        assert(mBiasWeights.type == DataType::kFLOAT || mBiasWeights.type == DataType::kHALF);

        mKernelWeights.values = malloc(mKernelWeights.count*type2size(mKernelWeights.type));
        memcpy(const_cast<void*>(mKernelWeights.values), weights[0].values, mKernelWeights.count*type2size(mKernelWeights.type));
        mBiasWeights.values = malloc(mBiasWeights.count*type2size(mBiasWeights.type));
        memcpy(const_cast<void*>(mBiasWeights.values), weights[1].values, mBiasWeights.count*type2size(mBiasWeights.type));

        mNbInputChannels = int(weights[0].count / nbOutputChannels /kernel_size_/ kernel_size_ * group);//根据输出元素的个数及核权重的个数计算输入的个数
    }

    // create the plugin at runtime from a byte stream
    //依据字节流创建接口
    ConvPlugin(const void* data, size_t length)
    {
        printf("2\n");
        const char* d = static_cast<const char*>(data), *a = d;//data转为字符串操作
        read(d, mNbInputChannels);//从d中读取输入元素个数
        read(d, mNbOutputChannels);//从d中读取输出元素个数
        read(d, mBiasWeights.count);
        read(d, mDataType);
        read(d,input_H_);
        read(d,input_W_);
        read(d,pad_);
        read(d,stride_);
        read(d,kernel_size_);
        read(d,dilation_);
        read(d,output_H_);
        read(d,output_W_);
        read(d,group_);
        mKernelWeights.count = mNbInputChannels * mNbOutputChannels * kernel_size_ * kernel_size_ / group_;//计算该层权重的个数
        
        mKernelWeights.values = nullptr;//设置权重值为空   
        mBiasWeights.values = nullptr;
        
        deserializeToDevice(d, mDeviceKernel, mKernelWeights.count*type2size(mDataType));
        deserializeToDevice(d, mDeviceBias, mBiasWeights.count*type2size(mDataType));
        
        assert(d == a + length);
    }

    ~ConvPlugin()
    {
        printf("3\n");
        if (mKernelWeights.values)
        {
            free(const_cast<void*>(mKernelWeights.values));
            mKernelWeights.values = nullptr;
        }
        if (mBiasWeights.values)
        {
            free(const_cast<void*>(mBiasWeights.values));
            mBiasWeights.values = nullptr;
        }
    }

    int getNbOutputs() const override
    {
        printf("4\n");
        return 1;//获取输出的样例数，这里等同于batchsize
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        printf("5\n");
        assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
        input_H_=inputs[0].d[1];
        input_W_=inputs[0].d[2];//获取输入层的高和
        
        output_H_=(input_H_-kernel_size_+2*pad_)/stride_+1;
        output_W_=(input_W_-kernel_size_+2*pad_)/stride_+1;
        
        return DimsCHW(mNbOutputChannels, output_H_, output_W_);
    }

    bool supportsFormat(DataType type, PluginFormat format) const override {
        printf("6\n");
    return (type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW; }
    
    

    void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override
    {
        printf("7\n");
        assert((type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW);
        mDataType = type;
    }





    int initialize() override//把偏置和网络模型权重copy到GPU
    {
        printf("8\n");
        CHECK(cudnnCreate(&mCudnn));// initialize cudnn and cublas
        (cudnnCreateTensorDescriptor(&bottom_desc_));// create cudnn tensor descriptors we need for bias addition
        (cudnnCreateTensorDescriptor(&top_desc_));
        (cudnnCreateTensorDescriptor(&bias_desc_));
        (cudnnCreateConvolutionDescriptor(&conv_desc_));
        (cudnnCreateFilterDescriptor(&filter_desc_));
        
        
        cudnnSetConvolution2dDescriptor(*&conv_desc_, pad_, pad_, stride_, stride_, dilation_, dilation_,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
      
        cudnnSetConvolutionGroupCount(*&conv_desc_, group_);
          
        
        (cudnnSetFilter4dDescriptor(*&filter_desc_, CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,mNbOutputChannels, mNbInputChannels / group_, kernel_size_, kernel_size_));
        
        (cudnnSetTensor4dDescriptor(*&bias_desc_, CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_FLOAT, 1,  mNbOutputChannels, 1, 1));
        
        if (mKernelWeights.values)
            convertAndCopyToDevice(mDeviceKernel, mKernelWeights);
        if (mBiasWeights.values)
            convertAndCopyToDevice(mDeviceBias, mBiasWeights);

        return 0;
    }

    virtual void terminate() override
    {
        printf("9\n");
        if (conv_desc_ != nullptr) 
        {
            cudnnDestroyConvolutionDescriptor(conv_desc_);
            conv_desc_ = nullptr;
        }
        if (bottom_desc_ != nullptr) 
        {
            cudnnDestroyTensorDescriptor(bottom_desc_);
            bottom_desc_ = nullptr;
        }
        if (top_desc_ != nullptr) 
        {
            cudnnDestroyTensorDescriptor(top_desc_);
            top_desc_ = nullptr;
        }
        if (filter_desc_ != nullptr) 
        {
            cudnnDestroyFilterDescriptor(filter_desc_);
            filter_desc_ = nullptr;
        }
        if (bias_desc_ != nullptr) 
        {
            cudnnDestroyTensorDescriptor(bias_desc_);
            bias_desc_ = nullptr;
        }
        if (mDeviceKernel)
        {
            cudaFree(mDeviceKernel);
            mDeviceKernel = nullptr;
        }
        if (mDeviceBias)
        {
            cudaFree(mDeviceBias);
            mDeviceBias = nullptr;
        }
    }

    virtual size_t getWorkspaceSize(int maxBatchSize) const override
    {
        printf("10\n");
        return 0;
    }

    virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override
    {
        printf("11\n");
        float oneval = 1.0;
        float zeroval = 0.0;
        const void* one = static_cast<void*>(&oneval);
        const void* zero = static_cast<void*>(&zeroval);

    
        (cudnnSetTensor4dDescriptor(*&bottom_desc_, CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_FLOAT, batchSize, mNbInputChannels, input_H_, input_W_));
        (cudnnSetTensor4dDescriptor(*&top_desc_, CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_FLOAT, batchSize, mNbOutputChannels, output_H_, output_W_));
        
        (cudnnConvolutionForward(mCudnn, one, bottom_desc_,
			inputs[0], filter_desc_,mDeviceKernel, conv_desc_,
			CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0,zero, top_desc_, outputs[0]));

        (cudnnAddTensor(mCudnn, one, bias_desc_,mDeviceBias, 
                one, top_desc_,outputs[0]));
        return 0;
    }
//序列化引擎阶段，把申请的所有私有变量包含到序列化里面
    virtual size_t getSerializationSize() override
    {
        printf("12\n");
        return sizeof(mNbInputChannels) + sizeof(mNbOutputChannels) + sizeof(mBiasWeights.count) + sizeof(mDataType) +
               (mKernelWeights.count + mBiasWeights.count) * type2size(mDataType) +sizeof(input_W_) +sizeof(input_H_)+ sizeof(output_H_)+ sizeof(output_W_) + sizeof(pad_) + sizeof(group_) + sizeof(pad_) + sizeof(dilation_) + sizeof(kernel_size_);
    }

    virtual void serialize(void* buffer) override
    {
        printf("13\n");
        char* d = static_cast<char*>(buffer), *a = d;
        write(d, mNbInputChannels);
        write(d, mNbOutputChannels);
        write(d, mBiasWeights.count);
        
        write(d, mDataType);
        write(d,input_H_);
        write(d,input_W_);
        write(d,pad_);
        write(d,stride_);
        write(d,kernel_size_);
        write(d,dilation_);
        write(d,output_H_);
        write(d,output_W_);
        write(d,group_);
        
        convertAndCopyToBuffer(d, mKernelWeights);
        convertAndCopyToBuffer(d, mBiasWeights);
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

    void* copyToDevice(const void* data, size_t count)
    {
        void* deviceData;
        CHECK(cudaMalloc(&deviceData, count));
        CHECK(cudaMemcpy(deviceData, data, count, cudaMemcpyHostToDevice));
        return deviceData;
    }

    void convertAndCopyToDevice(void*& deviceWeights, const Weights& weights)
    {
        if (weights.type != mDataType) // Weights are converted in host memory first, if the type does not match
        {
            size_t size = weights.count*(mDataType == DataType::kFLOAT ? sizeof(float) : sizeof(__half));
            void* buffer = malloc(size);
            for (int64_t v = 0; v < weights.count; ++v)
                if (mDataType == DataType::kFLOAT)
                    static_cast<float*>(buffer)[v] = fp16::__half2float(static_cast<const __half*>(weights.values)[v]);
                else
                    static_cast<__half*>(buffer)[v] = fp16::__float2half(static_cast<const float*>(weights.values)[v]);

            deviceWeights = copyToDevice(buffer, size);
            free(buffer);
        }
        else
            deviceWeights = copyToDevice(weights.values, weights.count * type2size(mDataType));
    }

    void convertAndCopyToBuffer(char*& buffer, const Weights& weights)
    {
        if (weights.type != mDataType)
            for (int64_t v = 0; v < weights.count; ++v)
                if (mDataType == DataType::kFLOAT)
                    reinterpret_cast<float*>(buffer)[v] = fp16::__half2float(static_cast<const __half*>(weights.values)[v]);
                else
                    reinterpret_cast<__half*>(buffer)[v] = fp16::__float2half(static_cast<const float*>(weights.values)[v]);
        else
            memcpy(buffer, weights.values, weights.count * type2size(mDataType));
        buffer += weights.count * type2size(mDataType);
    }

    void deserializeToDevice(const char*& hostBuffer, void*& deviceWeights, size_t size)
    //从hostbuffer中拷贝size大小的数据给deviceweights
    {
        deviceWeights = copyToDevice(hostBuffer, size);
        hostBuffer += size;
    }


    int kernel_size_;
    int stride_;
    int pad_;
    int group_;
	int dilation_;
    int mNbOutputChannels, mNbInputChannels;
    int input_H_;
    int input_W_;
    int output_H_;
    int output_W_;
    Weights mKernelWeights, mBiasWeights;

    DataType mDataType{DataType::kFLOAT};
    void* mDeviceKernel{nullptr};
    void* mDeviceBias{nullptr};

    cudnnHandle_t mCudnn;
    cudnnConvolutionDescriptor_t conv_desc_ = nullptr;
    cudnnTensorDescriptor_t bottom_desc_ = nullptr, top_desc_ = nullptr;
    cudnnFilterDescriptor_t filter_desc_ = nullptr;
    cudnnTensorDescriptor_t bias_desc_ = nullptr;

};
