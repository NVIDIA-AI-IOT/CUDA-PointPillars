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

#import mayavi.mlab as mlab
import torch
from numpy import *
import numpy as np

from onnxsim import simplify
import onnx
import onnx_graphsurgeon as gs

#print("Graph.fold_constants Help:\n{}".format(gs.Graph.fold_constants.__doc__))

@gs.Graph.register()
def replace_with_clip(self, inputs, outputs):
    # Disconnect output nodes of all input tensors
    for inp in inputs:
        inp.outputs.clear()

    # Disconnet input nodes of all output tensors
    for out in outputs:
        out.inputs.clear()

    # Insert the new node.
    return self.layer(op="ScatterBEV", inputs=inputs, outputs=outputs)

def simplify_onnx(onnx_model):
  print("Use onnx_graphsurgeon to modify onnx...")
  # Now we'll do the actual replacement
  graph = gs.import_onnx(onnx_model)

  tmap = graph.tensors()
  tmp_inputs = graph.inputs

  MAX_VOXELS = tmap["input"].shape[0]
  #print(tmap["input"].shape[0])

  input_new = gs.Variable(name="input", dtype=np.float32, shape=(MAX_VOXELS, 32, 10))
  X = gs.Variable(name="coords", dtype=np.float32, shape=(1, 1, MAX_VOXELS, 4))
  Y = gs.Variable(name="params", dtype=np.float32, shape=(1, 1, 1, 5))

  first_node_after_pillarscatter = [node for node in graph.nodes if node.op == "Conv"][0]

  first_node_pillarvfe = [node for node in graph.nodes if node.op == "MatMul"][0]

  next_node = current_node = first_node_pillarvfe
  for i in range(3):
    next_node = [node for node in graph.nodes if node.inputs[0] == current_node.outputs[0]][0]
    current_node = next_node
    #print(next_node)

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

  # Notice that we do not need to manually modify the rest of the graph. ONNX GraphSurgeon will
  # take care of removing any unnecessary nodes or tensors, so that we are left with only the subgraph.
  graph.cleanup()
  # That's it!

  #Remane the first tensor for the first layer 

  graph.inputs = [input_new, X, Y]
  first_add = [node for node in graph.nodes if node.op == "MatMul"][0]
  first_add.inputs[0] = input_new

  graph.cleanup().toposort()

  return gs.export_onnx(graph)

if __name__ == '__main__':
    mode_file = "pointpillar-native-sim.onnx"
    simplify_onnx(onnx.load(mode_file))

