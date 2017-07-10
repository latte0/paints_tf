from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os


def create_op(func, **placeholders):
    op = func(**placeholders)

    def f(**kwargs):
        feed_dict = {}
        for argname, argvalue in kwargs.items():
            placeholder = placeholders[argname]
            feed_dict[placeholder] = argvalue
        return tf.get_default_session().run(op, feed_dict=feed_dict)

    return f

downscale = create_op(
    func=tf.image.resize_images,
    images=tf.placeholder(tf.float32, [None, None, None]),
    size=tf.placeholder(tf.int32, [2]),
    method=tf.image.ResizeMethod.AREA,
)

upscale = create_op(
    func=tf.image.resize_images,
    images=tf.placeholder(tf.float32, [None, None, None]),
    size=tf.placeholder(tf.int32, [2]),
    method=tf.image.ResizeMethod.BICUBIC,
)

decode_jpeg = create_op(
    func=tf.image.decode_jpeg,
    contents=tf.placeholder(tf.string),
)

decode_png = create_op(
    func=tf.image.decode_png,
    contents=tf.placeholder(tf.string),
)

rgb_to_grayscale = create_op(
    func=tf.image.rgb_to_grayscale,
    images=tf.placeholder(tf.float32),
)

grayscale_to_rgb = create_op(
    func=tf.image.grayscale_to_rgb,
    images=tf.placeholder(tf.float32),
)

encode_jpeg = create_op(
    func=tf.image.encode_jpeg,
    image=tf.placeholder(tf.uint8),
)

encode_png = create_op(
    func=tf.image.encode_png,
    image=tf.placeholder(tf.uint8),
)

crop = create_op(
    func=tf.image.crop_to_bounding_box,
    image=tf.placeholder(tf.float32),
    offset_height=tf.placeholder(tf.int32, []),
    offset_width=tf.placeholder(tf.int32, []),
    target_height=tf.placeholder(tf.int32, []),
    target_width=tf.placeholder(tf.int32, []),
)

pad = create_op(
    func=tf.image.pad_to_bounding_box,
    image=tf.placeholder(tf.float32),
    offset_height=tf.placeholder(tf.int32, []),
    offset_width=tf.placeholder(tf.int32, []),
    target_height=tf.placeholder(tf.int32, []),
    target_width=tf.placeholder(tf.int32, []),
)

to_uint8 = create_op(
    func=tf.image.convert_image_dtype,
    image=tf.placeholder(tf.float32),
    dtype=tf.uint8,
    saturate=True,
)

to_float32 = create_op(
    func=tf.image.convert_image_dtype,
    image=tf.placeholder(tf.uint8),
    dtype=tf.float32,
)

