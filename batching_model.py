import onnx

onnx_model = onnx.load('./model/retinaface_r50_v1_900x900.onnx')
onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '?' # dim_value = -1
for i in range(len(onnx_model.graph.input)):
    print(onnx_model.graph.input[i])

# Uncomment this line if you want to change outputs
# for i in range(len(onnx_model.graph.output)):
#     print('-'*60)
#     print(onnx_model.graph.output[i])
#     onnx_model.graph.output[i].type.tensor_type.shape.dim[0].dim_param = '?' # dim_value = -1
#     print(onnx_model.graph.output[i])

#     # Changing the names of endpoints
#     #if onnx_model.graph.output[i].name in endpoint_names:
#     #    print('-'*60)
#     #    print(onnx_model.graph.output[i])
#     #    onnx_model.graph.output[i].name = onnx_model.graph.output[i].name.split(':')[0]

onnx.save(onnx_model, './model/900_900_dynamic_model_output.onnx')
print('Finished.')