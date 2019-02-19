# Azure Custom Vision TensorFlow.js

The project shows how to use a model exported from Azure Custom Vision with TensorFlow.js and run on browser.

## Train the model with Azure Custom Vision

## Convert Frozen Model to a web-friendly format

```bash
tensorflowjs_converter --input_format=tf_frozen_model --output_node_names='model_outputs' saved_model/model.pb web_model
```

## Create a tf.Tensor from an image using tf.fromPixels

```js
const image = document.getElementById('image');
let imageObj = await tf.fromPixels(image);
```

## Send image (binary) to API 
https://developer.mozilla.org/en-US/docs/Web/API/FileReader/readAsArrayBuffer

## Deploy to AWS S3 Bucket

Remember to set CORS configuration

```xml
<?xml version="1.0" encoding="UTF-8"?>
<CORSConfiguration xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
<CORSRule>
    <AllowedOrigin>URL to your site</AllowedOrigin>
    <AllowedMethod>GET</AllowedMethod>
    <AllowedMethod>HEAD</AllowedMethod>
    <MaxAgeSeconds>3000</MaxAgeSeconds>
    <AllowedHeader>Authorization</AllowedHeader>
</CORSRule>
</CORSConfiguration>
```

## Reference

https://thekevinscott.com/image-classification-with-javascript/

https://stackoverflow.com/questions/32556664/getting-byte-array-through-input-type-file/32556944

http://blog.brew.com.hk/working-with-files-in-javascript/

https://thekevinscott.com/image-classification-with-javascript/

https://github.com/tensorflow/tfjs-converter

https://github.com/tensorflow/tfjs-converter/tree/master/demo/mobilenet

https://github.com/tensorflow/tfjs-examples/blob/master/webcam-transfer-learning/index.js

https://blog.mgechev.com/2018/10/20/transfer-learning-tensorflow-js-data-augmentation-mobile-net/

https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/drawImage

https://docs.aws.amazon.com/AmazonS3/latest/user-guide/add-cors-configuration.html

https://gist.github.com/philfreo/4695840

