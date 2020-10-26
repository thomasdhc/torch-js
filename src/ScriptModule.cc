#include "ScriptModule.h"

#include "Tensor.h"
#include "utils.h"

namespace torchjs {

Napi::Object ScriptModule::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func =
      DefineClass(env, "ScriptModule",
                  {
                      InstanceMethod("forward", &ScriptModule::forward),
                      InstanceMethod("forwardBertClassification", &ScriptModule::
forwardBertClassification),
                      InstanceMethod("forwardClassification", &ScriptModule::forw
ardClassification),
                      InstanceMethod("forwardDetection", &ScriptModule::forwardDe
tection)
                      InstanceMethod("toString", &ScriptModule::toString),
                  });

  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();
  exports.Set("ScriptModule", func);
  return exports;
}

ScriptModule::ScriptModule(const Napi::CallbackInfo &info)
    : Napi::ObjectWrap<ScriptModule>(info) {
  Napi::HandleScope scope(info.Env());
  Napi::String value = info[0].As<Napi::String>();
  path_ = value;
  torch::jit::setGraphExecutorOptimize(false);
  module_ = torch::jit::load(value);
  torch::Device device(torch::kCUDA);
  module_.to(torch::kHalf);
  module_.to(device);
  module_.eval();

  // set number of cpus to use
  at::init_num_threads();
  at::set_num_threads(1);
}

Napi::FunctionReference ScriptModule::constructor;

Napi::Value ScriptModule::forward(const Napi::CallbackInfo &info) {
  auto len = info.Length();

  // Convert Node JS buffer to string
  std::vector<torch::jit::IValue> inputs;
  auto fileBuffer = info[0].As<Napi::Buffer<char>>();
  auto array_len = fileBuffer.ElementLength();
  std::vector<uint8_t> dataVector;
  for (decltype(array_len) i = 0; i < array_len; ++i) {
    dataVector.push_back(fileBuffer[i]);
  }
  std::string fileBufferString(dataVector.begin(), dataVector.end());
  inputs.push_back(fileBufferString);

  torch::IValue pred = module_.forward(inputs);

  auto detections = pred.toTuple()->elements()[0];

  Napi::Env env = info.Env();
  auto obj = Napi::Object::New(env);

  if (detections.isTensor()) {
    auto outputs = detections.toTensor()[0];
    torch::Tensor boxes = outputs.slice(1, 0, 4).to(torch::kCPU).to(torch::kFloat32);
    torch::Tensor conf = outputs.select(1, 4).to(torch::kCPU).to(torch::kFloat32);
    torch::Tensor cls = outputs.select(1, 5).to(torch::kCPU).to(torch::kInt32);

    auto names = pred.toTuple()->elements()[1].toList();


    Napi::Array classes_array = Napi::Array::New(env, cls.numel());
    auto boxes_array = Napi::TypedArrayOf<float>::New(env, boxes.numel());
    auto scores_array = Napi:: TypedArrayOf<float>::New(env, conf.numel());

    // Get class from list of names
    for ( int i = 0; i < cls.numel(); i++) {
        Napi::String name = Napi::String::New(env, names.get(cls[i].item<int>()).toStringRef());
        classes_array[i] = name;
    }

    memcpy(boxes_array.Data(), boxes.data_ptr(), sizeof(float) * boxes.numel());
    memcpy(scores_array.Data(), conf.data_ptr(), sizeof(float) * conf.numel());

    obj.Set("classes", classes_array);
    obj.Set("scores", scores_array);
    obj.Set("boxes", boxes_array);
  } else {
    Napi::Array classes_array = Napi::Array::New(env, 0);
    auto boxes_array = Napi::TypedArrayOf<float>::New(env, 0);
    auto scores_array = Napi:: TypedArrayOf<float>::New(env, 0);
    obj.Set("classes", classes_array);
    obj.Set("scores", scores_array);
    obj.Set("boxes", boxes_array);
  }
  return obj;
}

Napi::Value ScriptModule::forwardBertClassification(const Napi::CallbackInfo &info) {
  auto len = info.Length();
  std::vector<torch::jit::IValue> inputs;
  for (size_t i = 0; i < len; ++i) {
    Tensor *tensor =
        Napi::ObjectWrap<Tensor>::Unwrap(info[i].As<Napi::Object>());
    inputs.push_back(tensor->tensor().to(torch::kCUDA));
  }

  auto outputs = module_.forward(inputs);
  torch::Tensor tensor = outputs.toTensor().to(torch::kCPU);
  Napi::Env env = info.Env();
  auto obj = Napi::Object::New(env);

  auto typed_array = Napi::TypedArrayOf<float>::New(env, tensor.numel());
  memcpy(typed_array.Data(), tensor.data_ptr(), sizeof(float) * tensor.numel());
  obj.Set("scores", typed_array);
  return obj;
}

Napi::Value ScriptModule::forwardClassification(const Napi::CallbackInfo &info) {

  std::vector<torch::jit::IValue> inputs;

  auto fileBuffer = info[0].As<Napi::Buffer<char>>();
  auto array_len = fileBuffer.ElementLength();
  std::vector<uint8_t> dataVector;
  for (decltype(array_len) i = 0; i < array_len; ++i) {
    dataVector.push_back(fileBuffer[i]);
  }
  std::string fileBufferString(dataVector.begin(), dataVector.end());
  inputs.push_back(fileBufferString);
  torch::IValue pred;

  pred = module_.forward(inputs);

  // Ignore batch for now
  auto label1 = pred.toTensorVector()[0];
  auto label2 = pred.toTensorVector()[1];

  auto score = torch::cat({label1, label2}, 1).to(torch::kCPU);

  Napi::Env env = info.Env();
  auto obj = Napi::Object::New(env);
  auto scores_array = Napi:: TypedArrayOf<float>::New(env, score.numel());

  memcpy(scores_array.Data(), score.data_ptr(), sizeof(float) * score.numel());
  obj.Set("scores", scores_array);
  return obj;
}

Napi::Value ScriptModule::forwardDetection(const Napi::CallbackInfo &info) {
  std::vector<torch::jit::IValue> inputs;

  auto fileBuffer = info[0].As<Napi::Buffer<char>>();
  auto array_len = fileBuffer.ElementLength();
  std::vector<uint8_t> dataVector;
  for (decltype(array_len) i = 0; i < array_len; ++i) {
    dataVector.push_back(fileBuffer[i]);
  }
  std::string fileBufferString(dataVector.begin(), dataVector.end());
  inputs.push_back(fileBufferString);

  torch::IValue pred = module_.forward(inputs);

  auto detections = pred.toTuple()->elements()[0];

  Napi::Env env = info.Env();
  auto obj = Napi::Object::New(env);

  if (detections.isTensor()) {
    auto outputs = detections.toTensor()[0];
    torch::Tensor boxes = outputs.slice(1, 0, 4).to(torch::kCPU).to(torch::kFloat32);
    torch::Tensor conf = outputs.select(1, 4).to(torch::kCPU).to(torch::kFloat32)

    torch::Tensor cls = outputs.select(1, 5).to(torch::kCPU).to(torch::kInt32);

    auto names = pred.toTuple()->elements()[1].toList();

    Napi::Array classes_array = Napi::Array::New(env, cls.numel());
    auto boxes_array = Napi::TypedArrayOf<float>::New(env, boxes.numel());
    auto scores_array = Napi:: TypedArrayOf<float>::New(env, conf.numel());

    // Get class from list of names
    for ( int i = 0; i < cls.numel(); i++) {
        Napi::String name = Napi::String::New(env, names.get(cls[i].item<int>()).toStringRef());
        classes_array[i] = name;
    }

    memcpy(boxes_array.Data(), boxes.data_ptr(), sizeof(float) * boxes.numel());
    memcpy(scores_array.Data(), conf.data_ptr(), sizeof(float) * conf.numel());

    obj.Set("classes", classes_array);
    obj.Set("scores", scores_array);
    obj.Set("boxes", boxes_array);
  } else {
    Napi::Array classes_array = Napi::Array::New(env, 0);
    auto boxes_array = Napi::TypedArrayOf<float>::New(env, 0);
    auto scores_array = Napi:: TypedArrayOf<float>::New(env, 0);
    obj.Set("classes", classes_array);
    obj.Set("scores", scores_array);
    obj.Set("boxes", boxes_array);
  }
  return obj;
}

Napi::Value ScriptModule::toString(const Napi::CallbackInfo &info) {
  return Napi::String::New(info.Env(), "ScriptModule(\"" + path_ + "\")");
}

} // namespace torchjs
