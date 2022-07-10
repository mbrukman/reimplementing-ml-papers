  // Begin: Inception module `module'

  format(`Inception_%s_Conv_1x1', module) [
    label = "Conv\n1x1+1(S)",
    class = "conv",
  ];

  format(`Inception_%s_Conv_1x1_reduce_3x3', module) [
    label = "Conv\n1x1+1(S)",
    class = "conv",
  ];

  format(`Inception_%s_Conv_3x3', module) [
    label = "Conv\n3x3+1(S)",
    class = "conv",
  ];

  format(`Inception_%s_Conv_1x1_reduce_5x5', module) [
    label = "Conv\n1x1+1(S)",
    class = "conv",
  ];

  format(`Inception_%s_Conv_5x5', module) [
    label = "Conv\n5x5+1(S)",
    class = "conv",
  ];

  format(`Inception_%s_MaxPool', module) [
    label = "MaxPool\n3x3+1(S)",
    class = "pool",
  ];

  format(`Inception_%s_MaxPool_Conv_1x1', module) [
    label = "Conv\n1x1+1(S)",
    class = "conv",
  ];

  format(`Inception_%s_DepthConcat', module) [
    label = "DepthConcat",
    class = "depthconcat",
  ];

  format(`Inception_%s_Conv_1x1', module) ->
    format(`Inception_%s_DepthConcat', module);
  format(`Inception_%s_Conv_1x1_reduce_3x3', module) ->
    format(`Inception_%s_Conv_3x3', module) ->
    format(`Inception_%s_DepthConcat', module);
  format(`Inception_%s_Conv_1x1_reduce_5x5', module) ->
    format(`Inception_%s_Conv_5x5', module) ->
    format(`Inception_%s_DepthConcat', module);
  format(`Inception_%s_MaxPool', module) ->
    format(`Inception_%s_MaxPool_Conv_1x1', module) ->
    format(`Inception_%s_DepthConcat', module);

  // End: Inception module `module'
