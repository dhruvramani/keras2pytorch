# Keras to PyTorch Model Converter

Converts your pretrained Keras models to Pytorch.

## Usage
Edit line number `10` and define your keras model. You can just copy the Sequential model here or import on your own.

Edit line number `46` to define the pytorch version of the model. This model has to be <b>exactly same</b> as your keras model. The order of layers, dimensions - exactly same.

Uncomment line number `94` and `108` to load your pretrained keras model and save the converted pytorch model. 

Call `convert2pytorch()` by passing the model paths.