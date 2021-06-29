import tensorflow as tf
import tensorflow.keras.layers as l
import numpy as np
import cv2


'''device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')'''

YOLOv4_tiny_anchors = [
    np.array([(5, 6), (11, 8), (7, 15)], np.float32),
    np.array([(20, 16), (14, 33), (37, 28)], np.float32),
    np.array([(56, 52), (96, 79), (140, 135)], np.float32),
]
classes = 73
ignore_thresh = 0.7
jitter = 0.3
mask = [6, 7, 8], [3, 4, 5], [0, 1, 2]
num = 9
random = 1
truth_thresh = 1
input_shape = 128


def yolo_nms(yolo_feats, yolo_max_boxes, yolo_iou_threshold, yolo_score_threshold):
    bbox_per_stage, objectness_per_stage, class_probs_per_stage = [], [], []

    for stage_feats in yolo_feats:
        num_boxes = (
            stage_feats[0].shape[1] * stage_feats[0].shape[2] * stage_feats[0].shape[3]
        )  # num_anchors * grid_x * grid_y
        bbox_per_stage.append(
            tf.reshape(
                stage_feats[0],
                (tf.shape(stage_feats[0])[0], num_boxes, stage_feats[0].shape[-1]),
            )
        )  # [None,num_boxes,4]
        objectness_per_stage.append(
            tf.reshape(
                stage_feats[1],
                (tf.shape(stage_feats[1])[0], num_boxes, stage_feats[1].shape[-1]),
            )
        )  # [None,num_boxes,1]
        class_probs_per_stage.append(
            tf.reshape(
                stage_feats[2],
                (tf.shape(stage_feats[2])[0], num_boxes, stage_feats[2].shape[-1]),
            )
        )  # [None,num_boxes,num_classes]

    bbox = tf.concat(bbox_per_stage, axis=1)
    objectness = tf.concat(objectness_per_stage, axis=1)
    class_probs = tf.concat(class_probs_per_stage, axis=1)

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.expand_dims(bbox, axis=2),
        scores=objectness * class_probs,
        max_output_size_per_class=yolo_max_boxes,
        max_total_size=yolo_max_boxes,
        iou_threshold=yolo_iou_threshold,
        score_threshold=yolo_score_threshold,
    )

    return [boxes, scores, classes, valid_detections]


def yolov3_boxes_regression(feats_per_stage, anchors_per_stage): # yolo1, YOLOv4_tiny_anchors[0]
    grid_size_x, grid_size_y = feats_per_stage.shape[1], feats_per_stage.shape[2]
    num_classes = feats_per_stage.shape[-1] - 5  # feats.shape[-1] = 4 + 1 + num_classes

    box_xy, box_wh, objectness, class_probs = tf.split(
        feats_per_stage, (2, 2, 1, num_classes), axis=-1
    )

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)

    grid = tf.meshgrid(tf.range(grid_size_y), tf.range(grid_size_x))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gy, gx, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.constant(
        [grid_size_y, grid_size_x], dtype=tf.float32
    )
    box_wh = tf.exp(box_wh) * anchors_per_stage

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs


def conv_classes_anchors(input, num_anchors_stage, num_classes):
    return tf.keras.layers.Reshape((input.shape[1], input.shape[2], num_anchors_stage, num_classes + 5))(input)


def conv(input, filters, padding, kernel_size, strides, batch_normalize, activation):
    if padding:
        padding = 'same'
    else:
        padding = 'valid'

    layer = l.Conv2D(filters=filters, padding=padding, kernel_size=kernel_size, strides=strides)(input)

    if batch_normalize:
        layer = l.BatchNormalization()(layer)

    if activation == 'leaky':
        layer = l.LeakyReLU()(layer)
    elif activation == 'linear':
        layer = tf.keras.activations.linear(layer)

    return layer


def CSP(input, fsize, rng):
    for i in range(rng):
        shortcut = input
        layer = conv(input, filters=fsize, padding=1, kernel_size=1, strides=1, batch_normalize=1, activation='leaky')
        layer = conv(layer, filters=fsize*2, padding=1, kernel_size=3, strides=1, batch_normalize=1, activation='leaky')
        layer = l.Add()([layer, shortcut])
        layer = tf.keras.activations.linear(layer)
        input = layer
    return layer


def YOLOv4_tiny(input_shape=128, num_classes=73, anchors=YOLOv4_tiny_anchors):
    inputs = tf.keras.Input(shape=(input_shape, input_shape, 3), dtype=tf.float32)
    layer = conv(inputs, filters=32, padding=1, kernel_size=3, strides=1, batch_normalize=1, activation='leaky')
    layer = conv(layer, filters=64, padding=1, kernel_size=3, strides=2, batch_normalize=1, activation='leaky')
    layer = CSP(layer, 32, 1)
    layer = conv(layer, filters=128, padding=1, kernel_size=3, strides=2, batch_normalize=1, activation='leaky')
    layer = CSP(layer, 64, 2)
    layer = conv(layer, filters=256, padding=1, kernel_size=3, strides=2, batch_normalize=1, activation='leaky')
    layer = CSP(layer, 128, 8)

    route1 = layer
    layer = conv(layer, filters=512, padding=1, kernel_size=3, strides=2, batch_normalize=1, activation='leaky')
    layer = CSP(layer, 256, 8)

    route2 = layer
    layer = conv(layer, filters=1024, padding=1, kernel_size=3, strides=2, batch_normalize=1, activation='leaky')
    layer = CSP(layer, 512, 4)

    layer = conv(layer, filters=512, padding=1, kernel_size=1, strides=1, batch_normalize=1, activation='leaky')
    layer = conv(layer, filters=1024, padding=1, kernel_size=3, strides=1, batch_normalize=1, activation='leaky')
    layer = conv(layer, filters=512, padding=1, kernel_size=1, strides=1, batch_normalize=1, activation='leaky')
    layer = conv(layer, filters=1024, padding=1, kernel_size=3, strides=1, batch_normalize=1, activation='leaky')
    layer = conv(layer, filters=512, padding=1, kernel_size=1, strides=1, batch_normalize=1, activation='leaky')
    route = layer

    layer = conv(layer, filters=1024, padding=1, kernel_size=3, strides=1, batch_normalize=1, activation='leaky')
    layer = conv(layer, filters=234, padding=1, kernel_size=1, strides=1, batch_normalize=0, activation='linear')
    # small objects
    yolo1 = conv_classes_anchors(layer, num_anchors_stage=len(anchors[0]), num_classes=num_classes)

    layer = conv(route, filters=256, padding=1, kernel_size=1, strides=1, batch_normalize=1, activation='leaky')
    layer = l.UpSampling2D(size=(2, 2))(layer)
    layer = tf.concat([layer, route2], axis=-1)  # route

    layer = conv(layer, filters=256, padding=1, kernel_size=1, strides=1, batch_normalize=1, activation='leaky')
    layer = conv(layer, filters=512, padding=1, kernel_size=3, strides=1, batch_normalize=1, activation='leaky')
    layer = conv(layer, filters=256, padding=1, kernel_size=1, strides=1, batch_normalize=1, activation='leaky')
    layer = conv(layer, filters=512, padding=1, kernel_size=3, strides=1, batch_normalize=1, activation='leaky')
    layer = conv(layer, filters=256, padding=1, kernel_size=1, strides=1, batch_normalize=1, activation='leaky')
    route = layer

    layer = conv(layer, filters=512, padding=1, kernel_size=3, strides=1, batch_normalize=1, activation='leaky')
    layer = conv(layer, filters=234, padding=1, kernel_size=1, strides=1, batch_normalize=0, activation='linear')
    # medium objects
    yolo2 = conv_classes_anchors(layer, num_anchors_stage=len(anchors[1]), num_classes=num_classes)

    layer = conv(route, filters=128, padding=1, kernel_size=1, strides=1, batch_normalize=1, activation='leaky')
    layer = l.UpSampling2D(size=(2, 2))(layer)
    layer = tf.concat([layer, route1], axis=-1)  # route

    layer = conv(layer, filters=128, padding=1, kernel_size=1, strides=1, batch_normalize=1, activation='leaky')
    layer = conv(layer, filters=256, padding=1, kernel_size=3, strides=1, batch_normalize=1, activation='leaky')
    layer = conv(layer, filters=128, padding=1, kernel_size=1, strides=1, batch_normalize=1, activation='leaky')
    layer = conv(layer, filters=256, padding=1, kernel_size=3, strides=1, batch_normalize=1, activation='leaky')
    layer = conv(layer, filters=128, padding=1, kernel_size=1, strides=1, batch_normalize=1, activation='leaky')
    layer = conv(layer, filters=256, padding=1, kernel_size=3, strides=1, batch_normalize=1, activation='leaky')
    layer = conv(layer, filters=234, padding=1, kernel_size=1, strides=1, batch_normalize=0, activation='linear')
    # big objects
    yolo3 = conv_classes_anchors(layer, num_anchors_stage=len(anchors[2]), num_classes=num_classes)

    predictions_1 = tf.keras.layers.Lambda(
        lambda x_input: yolov3_boxes_regression(x_input, anchors[0]),
        name="yolov3_boxes_regression_small_scale",
    )(yolo1)
    predictions_2 = tf.keras.layers.Lambda(
        lambda x_input: yolov3_boxes_regression(x_input, anchors[1]),
        name="yolov3_boxes_regression_medium_scale",
    )(yolo2)
    predictions_3 = tf.keras.layers.Lambda(
        lambda x_input: yolov3_boxes_regression(x_input, anchors[2]),
        name="yolov3_boxes_regression_large_scale",
    )(yolo3)

    outputs = tf.keras.layers.Lambda(
        lambda x_input: yolo_nms(
            x_input,
            yolo_max_boxes=10,
            yolo_iou_threshold=0.1,
            yolo_score_threshold=0.1,
        ),
        name="yolov4_nms",
    )([predictions_1, predictions_2, predictions_3])

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='YOLOv4_tiny')

    return model


input_data = np.random.randn(1, 128, 128, 3)
model = YOLOv4_tiny()
path = '/home/dmitriy/Desktop/YOLO_v4/darknet/data/signs3/backup/signs_last.weights'

#model.summary()

conv_layers = []
for i in range(75):
    if i == 0:
        conv_layers.append(model.get_layer('conv2d'))
    else:
        conv_layers.append(model.get_layer('conv2d_'+str(i)))

batch_norm_layers = []
for i in range(72):
    if i == 0:
        batch_norm_layers.append(model.get_layer('batch_normalization'))
    else:
        batch_norm_layers.append(model.get_layer('batch_normalization_'+str(i)))

# Open darknet file and read headers
darknet_weight_file = open(path, "rb")
# First elements of file are major, minor, revision, seen, _
_ = np.fromfile(darknet_weight_file, dtype=np.int32, count=5)


# Keep an index of which batch norm should be considered.
# If batch norm is used with a convolution (meaning conv has no bias), the index is incremented
# Otherwise (conv has a bias), index is kept still.
current_matching_batch_norm_index = 0
a = 0
for layer in conv_layers:
    kernel_size = layer.kernel_size
    input_filters = layer.input_shape[-1]
    filters = layer.filters
    use_bias = layer.bias is not None

    if use_bias:
        # Read bias weight
        conv_bias = np.fromfile(
            darknet_weight_file, dtype=np.float32, count=filters
        )
        a += len(conv_bias)
    else:
        # Read batch norm
        # Reorder from darknet (beta, gamma, mean, var) to TF (gamma, beta, mean, var)
        batch_norm_weights = np.fromfile(
            darknet_weight_file, dtype=np.float32, count=4 * filters
        ).reshape((4, filters))[[1, 0, 2, 3]]

    # Read kernel weights
    # Reorder from darknet (filters, input_filters, kernel_size[0], kernel_size[1]) to
    # TF (kernel_size[0], kernel_size[1], input_filters, filters)
    conv_size = kernel_size[0] * kernel_size[1] * input_filters * filters
    conv_weights = (
        np.fromfile(darknet_weight_file, dtype=np.float32, count=conv_size)
        .reshape((filters, input_filters, kernel_size[0], kernel_size[1]))
        .transpose([2, 3, 1, 0])
    )

    if use_bias:
        # load conv weights and bias, increase batch_norm offset
        layer.set_weights([conv_weights, conv_bias])
    else:
        # load conv weights, load batch norm weights
        layer.set_weights([conv_weights])
        batch_norm_layers[current_matching_batch_norm_index].set_weights(
            batch_norm_weights
        )
        current_matching_batch_norm_index += 1

# NBSPCheck if we read the entire darknet file.
remaining_chars = len(darknet_weight_file.read())
darknet_weight_file.close()
#print(remaining_chars)
'''assert remaining_chars == 0
a = model.get_weights()
b = model.layers

for i in range(len(a)):
    print(len(a[i]), b[i])'''



#model.save('yolov4_tiny_weights')
#model = tf.keras.models.load_model('yolov4_tiny_weights')


#img = '/home/dmitriy/all_datasets/signs/train/5.19/_185.10.jpg'
img = '/home/dmitriy/Desktop/YOLO_v4/darknet/crop.png'
img = cv2.imread(img)
img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
img = img.reshape(1, 128, 128, 3)
with tf.device('/device:GPU:0'):
    pred = model.predict(img)
boxes = pred[0][0]
percents = pred[1][0]
classes = pred[2][0]
print('boxes:', boxes)
print('percents: ', percents)
print('classes: ', classes)

img = img.reshape(128, 128, 3)
for i in range(len(boxes)-1):
    if percents[i] > 0.6:

        x = boxes[i][0]
        y = boxes[i][1]
        w = boxes[i][2]
        h = boxes[i][3]

        x1 = int((x-w/2)*128)
        y1 = int((y-h/2)*128)
        x2 = int((x+w/2)*128)
        y2 = int((y+h/2)*128)

        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
cv2.imshow('', img)
cv2.waitKey(0)



