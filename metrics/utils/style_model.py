import tensorflow as tf


def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    # Retrieve the intermediate outputs of the layers corresponding to the names
    outputs = [vgg.get_layer(name).output for name in layer_names]

    # Create a model that maps VGG inputs to the intermediate outputs (it returns a list of intermediate outputs)
    model = tf.keras.Model([vgg.input], outputs)
    return model


# Function to compute the Gram matrix of an input tensor
def gram_matrix(input_tensor):
    # Compute the Gram matrix using Einstein summation
    """
    Multiplication of feature maps by themselves transposed,
    which generates a matrix representing the correlations between different feature maps

        b: axis related to batch size
        i: axis relative to the height of the feature maps
        j: axis relative to the width of the feature maps
        c: axis related to the number of channels in the feature maps of the first tensor
        d: axis related to the number of channels in the feature maps of the second tensor
    """
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)

    # the size of each matrix element is divided by the number of positions in the feature map space (width x height),
    # ensuring that the scale of the matrix does not depend on the size of the input image.
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


# Class which combines style and content information
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        # Create a VGG model with the specified style and content layers
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):  # 0-255
        # Preprocess the input image using VGG19 preprocessing (pixel normalization)
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)

        # Pass the preprocessed input through the VGG model
        outputs = self.vgg(preprocessed_input)

        # Split the outputs into style and content layers
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])

        # Compute Gram matrices for every layers of style outputs
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        # Create dictionaries that associates each style/content layer name with their respective Gram matrices
        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}
