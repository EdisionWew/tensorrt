# tensorrt
tensorrt 的第一步是定义模型网络，可以选择从tensorrt的语法库中导入模型网络例如caffe、uff( used for tensorflow)、ONNX，也可以选择利用tensorrt API自定义网络

C++
1.使用解析器引入模型网络
	a.创建一个tensorrt builder和网络(顺序不能反)
	b.创建一个具体格式的tensorrt parser
	c.使用创建的解析器parser解析引入网络并移植
	
	ex. Import caffe model
		IBuilder* builder = createInferBuilder(gLogger);
	       INetworkDefinition* network = builder->createNetwork()//创建builder和network
	       ICaffeParser* parser = createCaffeParser();//创建解析器
	      const IBlobNameToTensor* blobNameToTensor = parser->parse("deploy_file" ,
 	      		"modelFile", *network, DataType::kFLOAT);//解析植入的网络

	import  tensorflow model
		IBuilder* builder = createInferBuilder(gLogger);
	      	INetworkDefinition* network = builder->createNetwork()//创建builder和network
		IUFFParser* parser = createUffParser();//创建解析器
		parser->registerInput("Input_0", DimsCHW(1, 28, 28), UffInputOrder::kNCHW);
		parser->registerOutput("Binary_3”);//声明网络的输入和输出
		parser->parse(uffFile, *network, nvinfer1::DataType::kFLOAT); //解析植入模型网络

	import  ONNX model

		nvonnxparser::IOnnxConfig* config = nvonnxparser::createONNXConfig();//Create Parser
		nvonnxparser::IONNXParser* parser = nvonnxparser::createONNXParser(*config);
		parser->parse(onnx_filename, DataType::kFLOAT); //解析ONNX模型
		parser->convertToTRTNetwork(); //转换模型成Tensorflow模型
		nvinfer1::INetworkDefinition* trtNetwork = parser->getTRTNetwork(); //从解析器中获取转换成的网络 
2.使用tensorrt API创建网络
		
	ex.使用api创建一个输入、卷积、pooling、全连接、激活、softmax的网络

		IBuilder* builder = createInferBuilder(gLogger);
		INetworkDefinition* network = builder->createNetwork();//创建builder和network
		auto data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{1, INPUT_H,INPUT_W});//创建网络Input
		
		layerName->getOutput(0)
		auto conv1 = network->addConvolution(*data->getOutput(0), 20, DimsHW{5, 5},weightMap["conv1filter"], weightMap["conv1bias"]);
		conv1->setStride(DimsHW{1, 1});//添加卷积层
		auto pool1 = network->addPooling(*conv1->getOutput(0), PoolingType::kMAX,DimsHW{2, 2});
		pool1->setStride(DimsHW{2, 2});//添加pool层
		auto ip1 = network->addFullyConnected(*pool1->getOutput(0), 500,weightMap["ip1filter"], weightMap["ip1bias”]);//添加全连接层
		auto relu1 = network->addActivation(*ip1->getOutput(0),ActivationType::kRELU);//添加rule层
		auto prob = network->addSoftMax(*relu1->getOutput(0));//添加softmax层
		prob->getOutput(0)->setName(OUTPUT_BLOB_NAME);

3.创建引擎

		builder->setMaxBatchSize(maxBatchSize);
		builder->setMaxWorkspaceSize(1 << 20);
		ICudaEngine* engine = builder->buildCudaEngine(*network);//创建引擎

4.序列化模型
		
		IHostMemory *serializedModel = engine->serialize();
		// store model to disk
		serializedModel->destroy();

		IRuntime* runtime = createInferRuntime(gLogger);
		ICudaEngine* engine = runtime->deserializeCudaEngine(modelData, modelSize,nullptr); //创建一个运行对象来反序列化
5.执行推理
		
		IExecutionContext *context = engine->createExecutionContext()；//上下文伴随着引擎建立，它存储了引擎所需要的模型参数、网络定义
		int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
		int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);//根据输入输出blob名字获取对应的输入输出索引
		void* buffers[2];
		buffers[inputIndex] = inputbuffer;
		buffers[outputIndex] = outputBuffer;//根据index在GPU开辟对应的buffer
		context.enqueue(batchSize, buffers, stream, nullptr);//将对应的buffer数据推入GPU steam队列 



