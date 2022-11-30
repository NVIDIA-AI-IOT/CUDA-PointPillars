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

import onnx
import numpy as np
import onnx_graphsurgeon as gs

@gs.Graph.register()
def replace_with_clip(self, inputs, outputs, voxel_array):
    for inp in inputs:
        inp.outputs.clear()

    for out in outputs:
        out.inputs.clear()

    op_attrs = dict()
    op_attrs["dense_shape"] = voxel_array

    return self.layer(name="PPScatter_0", op="PPScatterPlugin", inputs=inputs, outputs=outputs, attrs=op_attrs)

def loop_node(graph, current_node, loop_time=0):
  for i in range(loop_time):
    next_node = [node for node in graph.nodes if len(node.inputs) != 0 and len(current_node.outputs) != 0 and node.inputs[0] == current_node.outputs[0]][0]
    current_node = next_node
  return next_node

def simplify_postprocess(onnx_model, FEATURE_SIZE_X, FEATURE_SIZE_Y, NUMBER_OF_CLASSES):
  print("Use onnx_graphsurgeon to adjust postprocessing part in the onnx...")
  graph = gs.import_onnx(onnx_model)

  cls_preds = gs.Variable(name="cls_preds", dtype=np.float32, shape=(1, int(FEATURE_SIZE_Y), int(FEATURE_SIZE_X), 2*NUMBER_OF_CLASSES*NUMBER_OF_CLASSES))
  box_preds = gs.Variable(name="box_preds", dtype=np.float32, shape=(1, int(FEATURE_SIZE_Y), int(FEATURE_SIZE_X), 14*NUMBER_OF_CLASSES))
  dir_cls_preds = gs.Variable(name="dir_cls_preds", dtype=np.float32, shape=(1, int(FEATURE_SIZE_Y), int(FEATURE_SIZE_X), 4*NUMBER_OF_CLASSES))

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


def simplify_preprocess(onnx_model, VOXEL_SIZE_Y, VOXEL_SIZE_X, MAX_POINTS_PER_VOXEL):
  print("Use onnx_graphsurgeon to modify onnx...")
  graph = gs.import_onnx(onnx_model)

  tmap = graph.tensors()
  MAX_VOXELS = tmap["voxels"].shape[0]

  VOXEL_ARRAY = np.array([int(VOXEL_SIZE_Y),int(VOXEL_SIZE_X)])

  # voxels: [V, P, C']
  # V is the maximum number of voxels per frame
  # P is the maximum number of points per voxel
  # C' is the number of channels(features) per point in voxels.
  input_new = gs.Variable(name="voxels", dtype=np.float32, shape=(MAX_VOXELS, MAX_POINTS_PER_VOXEL, 10))

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
  graph.replace_with_clip(inputs, outputs, VOXEL_ARRAY)

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

if __name__ == '__main__':
    mode_file = "pointpillar-native-sim.onnx"
    simplify_preprocess(onnx.load(mode_file))
