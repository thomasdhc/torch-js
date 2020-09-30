#include "Tensor.h"

#include "constants.h"
#include "utils.h"

namespace torchjs {

using namespace constants;

namespace {
template <typename T>
Napi::Value tensorToArray(Napi::Env env, const torch::Tensor &tensor) {
  Napi::EscapableHandleScope scope(env);
  assert(tensor.is_contiguous());
  auto typed_array = Napi::TypedArrayOf<T>::New(env, tensor.numel());
  memcpy(typed_array.Data(), tensor.data_ptr(), sizeof(T) * tensor.numel());
  auto shape_array = tensorShapeToArray(env, tensor);
  auto obj = Napi::Object::New(env);
  obj.Set(kData, typed_array);
  obj.Set(kShape, shape_array);
  return scope.Escape(obj);
}

template <typename T>
Napi::Value arrayToTensor(Napi::Env env, const Napi::TypedArray &data,
                          const ShapeArrayType &shape_array) {
  Napi::EscapableHandleScope scope(env);
  auto *data_ptr = data.As<Napi::TypedArrayOf<T>>().Data();
  auto shape = shapeArrayToVector(shape_array);

  auto dataVector = napiArrayToVector(data);
  // TODO add better support for kInt64
  // Currently casting all tensors to kInt64
  auto options = torch::TensorOptions()
    .dtype(torch::kInt64);
  auto torch_tensor = torch::from_blob(dataVector.data(), shape, options);
  torch::Device device(torch::kCUDA);
  torch_tensor = torch_tensor.to(device);
  return scope.Escape(Tensor::FromTensor(env, torch_tensor));
}
} // namespace

Napi::Object Tensor::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func =
      DefineClass(env, "Tensor",
                  {
                      InstanceMethod("toString", &Tensor::toString),
                      InstanceMethod("toObject", &Tensor::toObject),
                      StaticMethod("fromImagePath", &Tensor::fromImagePath),
                      StaticMethod("fromObject", &Tensor::fromObject),
                  });

  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();
  exports.Set("Tensor", func);
  return exports;
}

Napi::Object Tensor::FromTensor(Napi::Env env, const torch::Tensor &tensor) {
  Napi::EscapableHandleScope scope(env);
  auto obj = constructor.New({});
  Napi::ObjectWrap<Tensor>::Unwrap(obj)->tensor_ = tensor;
  return scope.Escape(obj).ToObject();
}

Tensor::Tensor(const Napi::CallbackInfo &info) : ObjectWrap<Tensor>(info) {}

Napi::FunctionReference Tensor::constructor;

torch::Tensor Tensor::tensor() { return tensor_; }

Napi::Value Tensor::toString(const Napi::CallbackInfo &info) {
  return Napi::String::New(info.Env(), tensor_.toString());
}

Napi::Value Tensor::toObject(const Napi::CallbackInfo &info) {
  auto env = info.Env();
  auto st = tensor_.scalar_type();
  switch (st) {
  case torch::ScalarType::Float:
    return tensorToArray<float>(env, tensor_);
  case torch::ScalarType::Double:
    return tensorToArray<double>(env, tensor_);
  case torch::ScalarType::Int:
    return tensorToArray<int32_t>(env, tensor_);
  case torch::ScalarType::Long:
    return tensorToArray<int32_t>(env, tensor_.to(torch::kInt32));
  default:
    throw Napi::TypeError::New(env, "Unsupported type");
  }
}

Napi::Value Tensor::fromObject(const Napi::CallbackInfo &info) {
  auto env = info.Env();
  // Is it my responsibility to deallocate these? Use a scope?
  auto obj = info[0].As<Napi::Object>();
  auto data = obj.Get(kData).As<Napi::TypedArray>();
  auto shape = obj.Get(kShape).As<ShapeArrayType>();
  auto dtype = obj.Get(kDtype).As<Napi::Number>().Int32Value();

  switch (dtype) {
  case static_cast<int32_t>(torch::kFloat32):
    return arrayToTensor<float>(env, data, shape);
  case static_cast<int32_t>(torch::kFloat64):
    return arrayToTensor<double>(env, data, shape);
  case static_cast<int32_t>(torch::kInt32):
    return arrayToTensor<int32_t>(env, data, shape);
  case static_cast<int32_t>(torch::kInt64):
    return arrayToTensor<int32_t>(env, data, shape);
  default:
    throw Napi::TypeError::New(env, "Unsupported type");
  }
}

Napi::Value Tensor::fromImagePath(const Napi::CallbackInfo &info) {
  auto env = info.Env();
  auto source = info[0].As<Napi::String>().string();
  cv::Mat img = cv::imread(source);
  if (img.empty()) {
    std::cerr << "Error loading the image!\n";
    return -1;
  }
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  img.convertTo(img, CV_32FC3, 1.0f / 255.0f);
  auto tensor_img = torch::from_blob(img.data, {1, img.rows, img.cols, img.channels()}).to(torch::kCUDA);
  tensor_img = tensor_img.permute({0, 3, 1, 2}).contiguous();
  tensor_img = tensor_img.to(torch::kHalf);
  return scope.Escape(Tensor::FromTensor(env, tensor_img));
}

} // namespace torchjs
