// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This is a recreation of the GoogLeNet model definition as it appears in the
// paper: https://arxiv.org/abs/1409.4842, but with each Inception module
// highlighted with a subcluster.
//
// See the Makefile to generate the SVG output; this file must be pre-processed
// with M4 first.

digraph GoogLeNet {

  label = "GoogLeNet model with Inception module groupings";
  fontsize = 32;
  labelloc = "top";
  rankdir = "BT";

  stylesheet = "googlenet.css";

  // Default settings for all nodes.
  node [
    shape = "box",
    fontname = "Verdana",
    fontsize = 10,
  ];

  input [
    shape = "octagon",
  ];

  Conv_7x7 [
    label = "Conv\n7x7+2(S)",
    class = "conv",
    fillcolor = "#346df1",
  ];

  MaxPool_1 [
    label = "MaxPool\n3x3+2(S)",
    class = "pool",
  ];

  LocalRespNorm_1 [
    label = "LocalRespNorm",
    class = "localrespnorm",
  ];

  Conv_1x1 [
    label = "Conv\n1x1+1(V)",
    class = "conv",
  ];

  Conv_3x3 [
    label = "Conv\n3x3+1(S)",
    class = "conv",
  ];

  LocalRespNorm_2 [
    label = "LocalRespNorm",
    class = "localrespnorm",
  ];

  MaxPool_2 [
    label = "MaxPool\n3x3+2(S)",
    class = "pool",
  ];

  MaxPool_3 [
    label = "MaxPool\n3x3+2(S)",
    class = "pool",
  ];

  MaxPool_4 [
    label = "MaxPool\n3x3+2(S)",
    class = "pool",
  ];

  // Output 0

  output0_AveragePool [
    label = "AveragePool\n5x5+3(V)",
    class = "pool",
  ];

  output0_Conv [
    label = "Conv\n1x1+1(S)",
    class = "conv",
  ];

  output0_FC_1 [
    label = "FC",
    class = "fc",
  ];

  output0_FC_2 [
    label = "FC",
    class = "fc",
  ];

  output0_Activation [
    label = "SoftmaxActivation",
    class = "softmax",
  ];

  softmax0 [
    shape = "octagon",
  ];

  // Output 1

  output1_AveragePool [
    label = "AveragePool\n5x5+3(V)",
    class = "pool",
  ];

  output1_Conv [
    label = "Conv\n1x1+1(S)",
    class = "conv",
  ];

  output1_FC_1 [
    label = "FC",
    class = "fc",
  ];

  output1_FC_2 [
    label = "FC",
    class = "fc",
  ];

  output1_Activation [
    label = "SoftmaxActivation",
    class = "softmax",
  ];

  softmax1 [
    shape = "octagon",
  ];

  // Output 2

  output2_AveragePool [
    label = "AveragePool\n5x5+3(V)",
    class = "pool",
  ];

  output2_FC [
    label = "FC",
    class = "fc",
  ];

  output2_Activation [
    label = "SoftmaxActivation",
    class = "softmax",
  ];

  softmax2 [
    shape = "octagon",
  ];

  // All the Inception modules.

  define(`module', 3a)
  include(`inception-cluster.m4')

  define(`module', 3b)
  include(`inception-cluster.m4')

  define(`module', 4a)
  include(`inception-cluster.m4')

  define(`module', 4b)
  include(`inception-cluster.m4')

  define(`module', 4c)
  include(`inception-cluster.m4')

  define(`module', 4d)
  include(`inception-cluster.m4')

  define(`module', 4e)
  include(`inception-cluster.m4')

  define(`module', 5a)
  include(`inception-cluster.m4')

  define(`module', 5b)
  include(`inception-cluster.m4')

  input ->
    Conv_7x7 ->
    MaxPool_1 ->
    LocalRespNorm_1 ->
    Conv_1x1 ->
    Conv_3x3 ->
    LocalRespNorm_2 ->
    MaxPool_2 -> {
      Inception_3a_Conv_1x1
      Inception_3a_Conv_1x1_reduce_3x3
      Inception_3a_Conv_1x1_reduce_5x5
      Inception_3a_MaxPool
    };  // connected to Inception_3a_DepthConcat above

  Inception_3a_DepthConcat -> {
    Inception_3b_Conv_1x1
    Inception_3b_Conv_1x1_reduce_3x3
    Inception_3b_Conv_1x1_reduce_5x5
    Inception_3b_MaxPool
  };  // connected to Inception_3b_DepthConcat above

  Inception_3b_DepthConcat -> MaxPool_3 -> {
    Inception_4a_Conv_1x1
    Inception_4a_Conv_1x1_reduce_3x3
    Inception_4a_Conv_1x1_reduce_5x5
    Inception_4a_MaxPool
  };  // connected to Inception_4a_DepthConcat above

  Inception_4a_DepthConcat -> {
    Inception_4b_Conv_1x1
    Inception_4b_Conv_1x1_reduce_3x3
    Inception_4b_Conv_1x1_reduce_5x5
    Inception_4b_MaxPool
  };  // connected to Inception_4b_DepthConcat above

  Inception_4b_DepthConcat -> {
    Inception_4c_Conv_1x1
    Inception_4c_Conv_1x1_reduce_3x3
    Inception_4c_Conv_1x1_reduce_5x5
    Inception_4c_MaxPool
  };  // connected to Inception_4c_DepthConcat above

  Inception_4c_DepthConcat -> {
    Inception_4d_Conv_1x1
    Inception_4d_Conv_1x1_reduce_3x3
    Inception_4d_Conv_1x1_reduce_5x5
    Inception_4d_MaxPool
  };  // connected to Inception_4d_DepthConcat above

  Inception_4d_DepthConcat -> {
    Inception_4e_Conv_1x1
    Inception_4e_Conv_1x1_reduce_3x3
    Inception_4e_Conv_1x1_reduce_5x5
    Inception_4e_MaxPool
  };  // connected to Inception_4e_DepthConcat above

  Inception_4e_DepthConcat -> MaxPool_4 -> {
    Inception_5a_Conv_1x1
    Inception_5a_Conv_1x1_reduce_3x3
    Inception_5a_Conv_1x1_reduce_5x5
    Inception_5a_MaxPool
  };  // connected to Inception_5a_DepthConcat above

  Inception_5a_DepthConcat -> {
    Inception_5b_Conv_1x1
    Inception_5b_Conv_1x1_reduce_3x3
    Inception_5b_Conv_1x1_reduce_5x5
    Inception_5b_MaxPool
  };  // connected to Inception_5b_DepthConcat above

  Inception_4a_DepthConcat ->
    output0_AveragePool ->
    output0_Conv ->
    output0_FC_1 ->
    output0_FC_2 ->
    output0_Activation ->
    softmax0;

  Inception_4d_DepthConcat ->
    output1_AveragePool ->
    output1_Conv ->
    output1_FC_1 ->
    output1_FC_2 ->
    output1_Activation ->
    softmax1;

  Inception_5b_DepthConcat ->
    output2_AveragePool ->
    output2_FC ->
    output2_Activation ->
    softmax2;
}
