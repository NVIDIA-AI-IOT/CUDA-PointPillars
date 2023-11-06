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
  print("Use onnx_graphsurgeon to modify onnx...")
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

  first_node_after_pillarscatter = [node for node in graph.nodes if node.op == "Conv"][0]

  first_node_pillarvfe = [node for node in graph.nodes if node.op == "MatMul"][0]

  next_node = current_node = first_node_pillarvfe
  for i in range(6):
    next_node = [node for node in graph.nodes if node.inputs[0] == current_node.outputs[0]][0]
    if i == 5:              # ReduceMax
      current_node.attrs['keepdims'] = [0]
      break
    current_node = next_node

  last_node_pillarvfe = current_node

  #merge some layers into one layer between inputs and outputs as below
  graph.inputs.append(Y)
  inputs = [last_node_pillarvfe.outputs[0], X, Y]
  outputs = [first_node_after_pillarscatter.inputs[0]]
  graph.replace_with_clip(inputs, outputs)

  # Remove the now-dangling subgraph.
  graph.cleanup().toposort()

  #just keep some layers between inputs and outputs as below
  graph.inputs = [first_node_pillarvfe.inputs[0] , X, Y]
  graph.outputs = [tmap["cls_preds"], tmap["box_preds"], tmap["dir_cls_preds"]]

  graph.cleanup()

  #Rename the first tensor for the first layer 
  graph.inputs = [input_new, X, Y]
  first_add = [node for node in graph.nodes if node.op == "MatMul"][0]
  first_add.inputs[0] = input_new

  graph.cleanup().toposort()

  return gs.export_onnx(graph)

def add_gather_to_onnx(onnx_model):

    graph = gs.import_onnx(onnx_model)
    rm_node = [node for node in graph.nodes if node.op == "ReduceMax"][-1]
    plugin_node = [node for node in graph.nodes if node.op == "PPScatterPlugin"][0]

    interm_var = gs.Variable(name="gather_in", shape=[10000, 64], dtype=np.float32)
    rm_node.outputs[0] = interm_var

    reshape_in = gs.Node(name="reshape_in_", op = "Reshape")
    reshape_in.inputs.append(interm_var)
    reshape_in_shape = gs.Constant(name="reshape_in_shape_", values = np.array([10000, 64, 1], dtype=np.int64))
    reshape_in.inputs.append(reshape_in_shape)
    reshape_in_out = gs.Variable(name="reshape_in_out_", shape = [10000, 64, 1], dtype=np.float32)
    reshape_in.outputs.append(reshape_in_out)
    graph.nodes.append(reshape_in)

    dummy_gather = gs.Node(name="dummy_gather_", op = "Gather")
    dummy_gather.inputs.append(reshape_in_out)
    dummy_gather.attrs['axis'] = 2
    dummy_gather_indices = gs.Constant(name="dummy_gather_indices_", values = np.array(0, dtype=np.int64))
    dummy_gather.inputs.append(dummy_gather_indices)
    dummy_gather_out = gs.Variable(name="dummy_gather_out_", shape = [10000, 64, 1], dtype=np.float32)
    dummy_gather.outputs.append(dummy_gather_out)
    graph.nodes.append(dummy_gather)

    reshape_out = gs.Node(name="reshape_out_", op = "Reshape")
    reshape_out.inputs.append(dummy_gather_out)
    reshape_out_shape = gs.Constant(name="reshape_out_shape_", values = np.array([10000, 64], dtype=np.int64))
    reshape_out.inputs.append(reshape_out_shape)
    reshape_out_out = gs.Variable(name="reshape_out_out_", shape = [10000, 64], dtype=np.float32)
    reshape_out.outputs.append(reshape_out_out)
    graph.nodes.append(reshape_out)

    plugin_node.inputs[0] = reshape_out_out

    return gs.export_onnx(graph)


if __name__ == '__main__':
    mode_file = "pointpillar-native-sim.onnx"
    simplify_preprocess(onnx.load(mode_file))
