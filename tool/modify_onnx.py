# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import onnx
import numpy as np
import onnx_graphsurgeon as gs

@gs.Graph.register()
def replace_with_clip(self, inputs, outputs):
    for inp in inputs:
        inp.outputs.clear()

    for out in outputs:
        out.inputs.clear()

    op_attrs = dict()
    op_attrs["dense_shape"] = np.array([496,432])

    return self.layer(name="PPScatter_0", op="PPScatterPlugin", inputs=inputs, outputs=outputs, attrs=op_attrs)

def loop_node(graph, current_node, loop_time=0):
  for i in range(loop_time):
    next_node = [node for node in graph.nodes if len(node.inputs) != 0 and len(current_node.outputs) != 0 and node.inputs[0] == current_node.outputs[0]][0]
    current_node = next_node
  return next_node

def simplify_postprocess(onnx_model):
  print("Use onnx_graphsurgeon to adjust postprocessing part in the onnx...")
  graph = gs.import_onnx(onnx_model)

  cls_preds = gs.Variable(name="cls_preds", dtype=np.float32, shape=(1, 248, 216, 18))
  box_preds = gs.Variable(name="box_preds", dtype=np.float32, shape=(1, 248, 216, 42))
  dir_cls_preds = gs.Variable(name="dir_cls_preds", dtype=np.float32, shape=(1, 248, 216, 12))

  tmap = graph.tensors()
  new_inputs = [tmap["voxels"], tmap["voxel_idxs"], tmap["voxel_num"]]
  new_outputs = [cls_preds, box_preds, dir_cls_preds]

  for inp in graph.inputs:
    if inp not in new_inputs:
      inp.outputs.clear()

  for out in graph.outputs:
    out.inputs.clear()

  first_ConvTranspose_node = [node for node in graph.nodes if node.op == "ConvTranspose"][0]
  concat_node = loop_node(graph, first_ConvTranspose_node, 3)
  assert concat_node.op == "Concat"

  first_node_after_concat = [node for node in graph.nodes if len(node.inputs) != 0 and len(concat_node.outputs) != 0 and node.inputs[0] == concat_node.outputs[0]]

  for i in range(3):
    transpose_node = loop_node(graph, first_node_after_concat[i], 1)
    assert transpose_node.op == "Transpose"
    transpose_node.outputs = [new_outputs[i]]

  graph.inputs = new_inputs
  graph.outputs = new_outputs
  graph.cleanup().toposort()
  
  return gs.export_onnx(graph)


def simplify_preprocess(onnx_model):
  print("Use onnx_graphsurgeon to adjust preprocessing part in the onnx...")
  graph = gs.import_onnx(onnx_model)

  tmap = graph.tensors()
  MAX_VOXELS = tmap["voxels"].shape[0]

  # voxels: [V, P, C']
  # V is the maximum number of voxels per frame
  # P is the maximum number of points per voxel
  # C' is the number of channels(features) per point in voxels.
  input_new = gs.Variable(name="voxels", dtype=np.float32, shape=(MAX_VOXELS, 32, 10))

  # voxel_idxs: [V, 4]
  # V is the maximum number of voxels per frame
  # 4 is just the length of indexs encoded as (frame_id, z, y, x).
  X = gs.Variable(name="voxel_idxs", dtype=np.int32, shape=(MAX_VOXELS, 4))

  # voxel_num: [1]
  # Gives valid voxels number for each frame
  Y = gs.Variable(name="voxel_num", dtype=np.int32, shape=(1,))

  reshape_0 = gs.Node(name="reshape_0", op = "Reshape")
  reshape_0.inputs.append(input_new)
  reshape_0_shape = gs.Constant(name="reshape_0_shape", values = np.array([MAX_VOXELS * 32, 10], dtype=np.int64))
  reshape_0.inputs.append(reshape_0_shape)
  reshape_0_out = gs.Variable(name="reshape_0_out", shape = [MAX_VOXELS * 32, 10], dtype=np.float32)
  reshape_0.outputs.append(reshape_0_out)
  graph.nodes.append(reshape_0)

  matmul_op = [node for node in graph.nodes if node.op == "MatMul"][0]
  matmul_op.inputs[0] = reshape_0_out
  matmul_op_out = gs.Variable(name="matmul_op_out", shape = [MAX_VOXELS * 32, 64], dtype=np.float32)
  matmul_op.outputs[0] = matmul_op_out

  bn_op = [node for node in graph.nodes if node.op == "BatchNormalization"][0]
  bn_op.inputs[0] = matmul_op_out
  bn_op_out = gs.Variable(name="bn_op_out", shape = [MAX_VOXELS * 32, 64], dtype=np.float32)
  bn_op.outputs[0] = bn_op_out

  relu_op = [node for node in graph.nodes if node.op == "Relu"][0]
  relu_op.inputs[0] = bn_op_out
  relu_op_out = gs.Variable(name="relu_op_out", shape = [MAX_VOXELS * 32, 64], dtype=np.float32)
  relu_op.outputs[0] = relu_op_out

  reshape_1 = gs.Node(name="reshape_1", op = "Reshape")
  reshape_1.inputs.append(relu_op_out)
  reshape_1_shape = gs.Constant(name="reshape_1_shape", values = np.array([MAX_VOXELS, 32, 64], dtype=np.int64))
  reshape_1.inputs.append(reshape_1_shape)
  reshape_1_out = gs.Variable(name="reshape_1_out", shape = [MAX_VOXELS, 32, 64], dtype=np.float32)
  reshape_1.outputs.append(reshape_1_out)
  graph.nodes.append(reshape_1)

  reducemax_op = [node for node in graph.nodes if node.op == "ReduceMax"][0]
  reducemax_op.inputs[0] = reshape_1_out
  reducemax_op.attrs['keepdims'] = [0]

  conv_op = [node for node in graph.nodes if node.op == "Conv"][0]
  graph.replace_with_clip([reducemax_op.outputs[0], X, Y], [conv_op.inputs[0]])

  graph.inputs = [input_new, X, Y]
  graph.outputs = [tmap["cls_preds"], tmap["box_preds"], tmap["dir_cls_preds"]]

  graph.cleanup().toposort()

  return gs.export_onnx(graph)


if __name__ == '__main__':
    mode_file = "pointpillar-native-sim.onnx"
    simplify_preprocess(onnx.load(mode_file))
