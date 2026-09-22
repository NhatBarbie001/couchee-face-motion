import onnx
import onnxruntime as ort
import numpy as np
from onnx import numpy_helper

model = onnx.load('models/checkpoints/sixdrepnet.onnx')
model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'batch'
model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = 'batch'

const_targets = [
    '/Constant_11_output_0', '/Constant_15_output_0', '/Constant_16_output_0',
    '/Constant_17_output_0', '/Constant_20_output_0', '/Constant_24_output_0',
    '/Constant_25_output_0', '/Constant_26_output_0'
]

count = 0
for n in model.graph.node:
    if n.op_type == 'Constant' and len(n.attribute) > 0 and n.attribute[0].t:
        if n.output[0] in const_targets:
            val = numpy_helper.to_array(n.attribute[0].t)
            if list(val) == [1, 1]:
                new_t = numpy_helper.from_array(np.array([-1, 1], dtype=np.int64), name=n.output[0])
                n.attribute[0].t.CopyFrom(new_t)
                count += 1

print(f"Patched {count} Constant nodes to [-1, 1]. Saving model...")
onnx.save(model, 'models/checkpoints/sixdrepnet.onnx')

print("Testing dynamic inference...")
sess = ort.InferenceSession('models/checkpoints/sixdrepnet.onnx', providers=['CPUExecutionProvider'])
dummy_1 = np.random.randn(1, 3, 224, 224).astype(np.float32)
dummy_4 = np.random.randn(4, 3, 224, 224).astype(np.float32)
dummy_32 = np.random.randn(32, 3, 224, 224).astype(np.float32)

out_1 = sess.run(None, {'input': dummy_1})[0]
out_4 = sess.run(None, {'input': dummy_4})[0]
out_32 = sess.run(None, {'input': dummy_32})[0]

print('SUCCESS!')
print('out_1 shape:', out_1.shape)
print('out_4 shape:', out_4.shape)
print('out_32 shape:', out_32.shape)
