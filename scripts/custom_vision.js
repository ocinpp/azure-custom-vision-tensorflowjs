const MODEL_URL = 'https://s3-ap-northeast-1.amazonaws.com/object-detect-demo-tf/tensorflowjs_model.pb';
const WEIGHTS_URL = 'https://s3-ap-northeast-1.amazonaws.com/object-detect-demo-tf/weights_manifest.json';
const ANCHORS = tf.tensor2d([[0.573, 0.677], [1.87, 2.06], [3.34, 5.47], [7.88, 3.53], [9.77, 9.17]]);
const IOU_THRESHOLD = 0.23;
const SCORE_THRESHOLD = 0.35;
const NORMALIZATION_OFFSET = 127.5;
const MAX_OUTPUT_SIZE = 20;

function drawBoundingBoxes(ctx, height, width, preds) {
  for (pred of preds) {
    drawBox(ctx, height, width, pred.probability, pred.boundingBox);
  }
}

function drawBox(ctx, height, width, probability, boundingBox) {
  let x = boundingBox.left * width
  let y = boundingBox.top * height
  let w = boundingBox.width * width
  let h = boundingBox.height * height

  ctx.rect(x, y, w, h);
  ctx.lineWidth = "2";
  ctx.strokeStyle = "lime";
  ctx.stroke();

  ctx.fillStyle = "navy";
  ctx.fillRect(x, y - 25, 85, 25);
  // ctx.fillRect(x, y - 25, 65, 25);

  ctx.font = "13pt arial";
  ctx.fillStyle = "white";
  ctx.fillText(Math.round(probability * 100, 0) + '%', x + 10, y - 5);

  // ctx.fillText('Man Utd', x + 8, y - 5);

}

// resize to 416 x 416
function resizeAndNormalize(img) {
  //h ttps://github.com/tensorflow/tfjs-models/blob/master/mobilenet/src/index.ts#L103-L114
  const normalized = img.toFloat();

  // Resize the image to
  let resized = normalized;
  if (img.shape[0] !== 416 || img.shape[1] !== 416) {
    const alignCorners = true;
    resized = tf.image.resizeBilinear(normalized, [416, 416], alignCorners);
  }

  return resized;
}

function extractBoundingBoxes(prediction_output, anchors) {
  const height = prediction_output.shape[1];
  const width = prediction_output.shape[2];
  const channels = prediction_output.shape[3];
  const num_anchor = anchors.shape[0]
  const num_class = Math.floor(channels / num_anchor) - 5

  console.log("num_anchor: " + num_anchor);
  console.log("num_class: " + num_class);

  // If one component of shape is the special value -1,
  // the size of that dimension is computed so that the
  // total size remains constant
  const outputs = prediction_output.reshape([height, width, num_anchor, -1]);

  console.log("outputs.shape: " + outputs.shape);

  const conv_dims = outputs.shape.slice(0, 2);
  const conv_dims_0 = conv_dims[0] // 13
  const conv_dims_1 = conv_dims[1] // 13

  // Extract bounding box information
  // x = (self._logistic(outputs[...,0]) + np.arange(width)[np.newaxis, :, np.newaxis]) / width
  // y = (self._logistic(outputs[...,1]) + np.arange(height)[:, np.newaxis, np.newaxis]) / height
  // w = np.exp(outputs[...,2]) * anchors[:,0][np.newaxis, np.newaxis, :] / width
  // h = np.exp(outputs[...,3]) * anchors[:,1][np.newaxis, np.newaxis, :] / height

  // reshape to 13, 13, 5
  const outputs_0 = outputs.slice([0, 0, 0, 0], [conv_dims_0, conv_dims_1, num_anchor, 1]).reshape([13,13,5]);
  const outputs_1 = outputs.slice([0, 0, 0, 1], [conv_dims_0, conv_dims_1, num_anchor, 1]).reshape([13,13,5]);
  const outputs_2 = outputs.slice([0, 0, 0, 2], [conv_dims_0, conv_dims_1, num_anchor, 1]).reshape([13,13,5]);
  const outputs_3 = outputs.slice([0, 0, 0, 3], [conv_dims_0, conv_dims_1, num_anchor, 1]).reshape([13,13,5]);
  const outputs_4 = outputs.slice([0, 0, 0, 4], [conv_dims_0, conv_dims_1, num_anchor, 1]).reshape([13,13,5]);
  const outputs_5 = outputs.slice([0, 0, 0, 5], [conv_dims_0, conv_dims_1, num_anchor, 1]);
  // console.log(_logistic(outputs_0));

  // https://js.tensorflow.org/api/0.6.1/#range
  // https://js.tensorflow.org/api/latest/index.html#tf.Tensor.expandDims

  let width_range = tf.range(0, width);
  width_range = width_range.expandDims(0).expandDims(2);

  let height_range = tf.range(0, height);
  height_range = height_range.expandDims(1).expandDims(2);

  // reshape to 1, 1, 5
  let anchors_0 = anchors.slice([0, 0], [num_anchor, 1]);
  anchors_0 = anchors_0.expandDims(0).expandDims(1).reshape([1,1,5]);

  // reshape to 1, 1, 5
  let anchors_1 = anchors.slice([0, 1], [num_anchor, 1]);
  anchors_1 = anchors_1.expandDims(0).expandDims(1).reshape([1,1,5]);

  // Extract bounding box information
  let x = tf.sigmoid(outputs_0).add(width_range).div(tf.scalar(width));
  let y = tf.sigmoid(outputs_1).add(height_range).div(tf.scalar(height));
  let w = tf.exp(outputs_2).mul(anchors_0).div(tf.scalar(width));
  let h = tf.exp(outputs_3).mul(anchors_1).div(tf.scalar(height));

  console.log(x.shape, y.shape, w.shape, h.shape);
  console.log(outputs_2.shape, outputs_3.shape, anchors_0.shape, anchors_1.shape);

  // (x,y) in the network outputs is the center of the bounding box. Convert them to top-left.
  x = x.sub(w.div(tf.scalar(2)));
  y = y.sub(h.div(tf.scalar(2)));

  // https://js.tensorflow.org/api/latest/index.html#stack
  boxes = tf.stack([x,y,w,h], axis=-1).reshape([-1, 4]);

  // [y1, x1, y2, x2] => top-left, right-bottom
  boxes_corners = tf.stack([y,x,y.sub(h),x.add(w)], axis=-1).reshape([-1, 4]);

  // Get confidence for the bounding boxes.
  const objectness = tf.sigmoid(outputs_4);

  // Get class probabilities for the bounding boxes.
  //const class_probs = tf.softmax(outputs_5, [conv_dims_0, conv_dims_1, num_anchor, num_class]);

  class_probs = outputs_5;

  const class_probs_max = tf.max(class_probs, 3);
  console.log(class_probs_max.shape);

  class_probs = tf.exp(class_probs.sub(class_probs_max.expandDims(3)));

  const class_probs_sum = tf.sum(class_probs, 3);
  console.log(class_probs_sum.shape);

  class_probs = class_probs.div(class_probs_sum.expandDims(3)).mul(objectness.expandDims(3))
  class_probs = class_probs.reshape([-1, num_class]);

  return [boxes, boxes_corners, class_probs];
}

async function postProcess(prediction_output) {
  const output = extractBoundingBoxes(prediction_output, ANCHORS);
  const output_boxes = output[0];
  const output_boxes_corners = output[1];
  const output_class_probs = output[2];

  // Performs non maximum suppression of bounding boxes based on iou (intersection over union)
  // https://js.tensorflow.org/api/0.12.5/#image.nonMaxSuppression
  // tf.image.nonMaxSuppression (boxes, scores, maxOutputSize, iouThreshold?, scoreThreshold?)
  const res = await tf.image.nonMaxSuppressionAsync(output_boxes_corners, output_class_probs.as1D(845),
                                              MAX_OUTPUT_SIZE, IOU_THRESHOLD, SCORE_THRESHOLD);
  const idx = res.dataSync();

  const res_box = output_boxes.gather(tf.tensor1d(idx, 'int32'));
  const res_prob = output_class_probs.gather(tf.tensor1d(idx, 'int32'));

  let boxes = [];
  let probs = [];

  tf.unstack(res_box).forEach(t => {
    boxes.push(t.dataSync());
  })

  tf.unstack(res_prob).forEach(t => {
    probs.push(t.dataSync());
  })

  // console.log(boxes);
  // console.log(probs);

  let result = [];
  let index = 0;

  for (prob of probs) {
    const obj = {'probability': prob[0],
                'boundingBox': {
                  'left': boxes[index][0],
                  'top': boxes[index][1],
                  'width': boxes[index][2],
                  'height': boxes[index][3]
                }};
    result.push(obj);
    index++;
  }

  return result;
}

async function loadModel() {
  const model = await tf.loadFrozenModel(MODEL_URL, WEIGHTS_URL);
  return model;
}

async function detectObject(model, ctx, img) {
  let imageObj = tf.fromPixels(img);

  imageObj = resizeAndNormalize(imageObj)
  // outputs = sess.run(output_tensor, {'Placeholder:0': inputs[np.newaxis,...]})

  // RGB -> BGR
  const channel_r = imageObj.slice([0, 0, 0], [416, 416, 1]);
  const channel_g = imageObj.slice([0, 0, 1], [416, 416, 1]);
  const channel_b = imageObj.slice([0, 0, 2], [416, 416, 1]);
  imageObj = tf.concat([channel_b, channel_g, channel_r], 2);

  // expand first dimension
  // https://js.tensorflow.org/api/latest/index.html#tf.Tensor.expandDims
  imageObj = imageObj.expandDims(0);

  const output = model.execute({Placeholder: imageObj});
  const result = await postProcess(output);

  console.log(result);

  drawBoundingBoxes(ctx, img.height, img.width, result);
}