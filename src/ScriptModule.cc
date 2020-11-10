#include "ScriptModule.h"

#include "Tensor.h"
#include "utils.h"

namespace torchjs {

Napi::Object ScriptModule::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func =
      DefineClass(env, "ScriptModule",
                  {
                      InstanceMethod("forward", &ScriptModule::forward),
                      InstanceMethod("forwardBertClassification", &ScriptModule::forwardBertClassification),
                      InstanceMethod("forwardClassification", &ScriptModule::forwardClassification),
                      InstanceMethod("forwardDetection", &ScriptModule::forwardDetection),
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

  at::init_num_threads();
  at::set_num_threads(1);
  torch::jit::setGraphExecutorOptimize(false);
  module_ = torch::jit::load(value);
  torch::Device device(torch::kCUDA);
  module_.to(device);
  // module_.to(torch::kHalf);
  module_.eval();
}

Napi::FunctionReference ScriptModule::constructor;

Napi::Value ScriptModule::forward(const Napi::CallbackInfo &info) {
  auto len = info.Length();
  std::vector<torch::jit::IValue> inputs;
  // TODO: Support other type of IValue, e.g., list
  for (size_t i = 0; i < len; ++i) {
    Tensor *tensor =
        Napi::ObjectWrap<Tensor>::Unwrap(info[i].As<Napi::Object>());
    inputs.push_back(tensor->tensor());
  }
  auto outputs = module_.forward(inputs);
  // TODO: Support other type of IValue
  assert(outputs.isTensor());
  torch::Tensor tensor = outputs.toTensor();
  Napi::Env env = info.Env();

  auto typed_array = Napi::TypedArrayOf<float>::New(env, tensor.numel());
  memcpy(typed_array.Data(), tensor.data_ptr(), sizeof(float) * tensor.numel());
  return typed_array;
  // return scope.Escape(Tensor::FromTensor(info.Env(), outputs.toTensor()));
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

  torch::NoGradGuard no_grad;

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

  // Hard coded threshold tensor
  auto thres = torch::tensor({0.6}).to(torch::kCUDA);
  inputs.push_back(thres);

  torch::IValue pred = module_.forward(inputs);

  auto bboxes = pred.toTuple()->elements()[0].toTensor().to(torch::kCPU).to(torch::kFloat32);
  auto scores = pred.toTuple()->elements()[1].toTensor().to(torch::kCPU).to(torch::kFloat32);
  auto classes = pred.toTuple()->elements()[2].toList().get(0).toList();

  Napi::Env env = info.Env();
  auto obj = Napi::Object::New(env);

  Napi::Array classes_array = Napi::Array::New(env, classes.size());
  auto bboxes_array = Napi::TypedArrayOf<float>::New(env, bboxes.numel());
  auto scores_array = Napi::TypedArrayOf<float>::New(env, scores.numel());

  for ( int i = 0; i < classes.size(); i++ ) {
    Napi::String class_name = Napi::String::New(env, classes.get(i).toStringRef());
    classes_array[i] = class_name;
  }

  memcpy(bboxes_array.Data(), bboxes.data_ptr(), sizeof(float) * bboxes.numel());
  memcpy(scores_array.Data(), scores.data_ptr(), sizeof(float) * scores.numel());

  obj.Set("classes", classes_array);
  obj.Set("scores", scores_array);
  obj.Set("bboxes", bboxes_array);

  return obj;
}

Napi::Value ScriptModule::toString(const Napi::CallbackInfo &info) {
  return Napi::String::New(info.Env(), "ScriptModule(\"" + path_ + "\")");
}

} // namespace torchjs
