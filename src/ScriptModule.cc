#include "ScriptModule.h"

#include "Tensor.h"
#include "utils.h"

namespace torchjs {

Napi::Object ScriptModule::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func =
      DefineClass(env, "ScriptModule",
                  {
                      InstanceMethod("forward", &ScriptModule::forward),
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
  std::vector<torch::jit::IValue> inputs;
  // TODO: Support other type of IValue, e.g., list
  torch::Device device(torch::kCUDA);
  auto fileBuffer = info[0].As<Napi::Buffer<char>>();
  auto array_len = fileBuffer.ElementLength();
  std::vector<uint8_t> dataVector;
  for (decltype(array_len) i = 0; i < array_len; ++i) {
    dataVector.push_back(fileBuffer[i]);
  }
  std::string fileBufferString(dataVector.begin(), dataVector.end());
  inputs.push_back(fileBufferString);
  // for (size_t i = 0; i < len; ++i) {
  //   Tensor *tensor =
  //       Napi::ObjectWrap<Tensor>::Unwrap(info[i].As<Napi::Object>());
  //   inputs.push_back(tensor->tensor());
  // }
  auto pred = module_.forward(inputs);

  torch::Tensor outputs = pred.toTuple()->elements()[0].toTensor();
  outputs = outputs.to(torch::kCPU);
  outputs = outputs.to(torch::kFloat32);
  Napi::Env env = info.Env();

  auto typed_array = Napi::TypedArrayOf<float>::New(env, outputs.numel());
  memcpy(typed_array.Data(), outputs.data_ptr(), sizeof(float) * outputs.numel());
  return typed_array;
}

Napi::Value ScriptModule::toString(const Napi::CallbackInfo &info) {
  return Napi::String::New(info.Env(), "ScriptModule(\"" + path_ + "\")");
}

} // namespace torchjs
