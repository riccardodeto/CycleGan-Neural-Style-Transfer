import tensorflow as tf


def random_crop(image, img_height, img_width, seed=123):
    cropped_image = tf.image.random_crop(image, size=[img_height, img_width, 3], seed=seed)
    return cropped_image

# normalizzazione dell'immagine a [-1, 1]
def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image

def random_jitter(image, seed=123):
    # resize dell'immagine a 286 x 286 x 3
    image = tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # rrandom crop a 256 x 256 x 3
    image = random_crop(image, 256, 256)
    # random mirroring
    image = tf.image.random_flip_left_right(image, seed=seed)
    return image

def preprocess_image_train(image, label):
    image = random_jitter(image)
    image = normalize(image)
    return image

def preprocess_image_test(image, label):
    image = normalize(image)
    return image

def preprocess_image_val(image, label):
    image = normalize(image)
    return image

def preprocess_dataset(train_portrait, val_portrait, test_portrait, train_naruto, val_naruto, test_naruto, buffer_size, batch_size, val_batch_size):
    # ottimizzazione dinamica per sfruttare al meglio le risorse disponibili
    AUTOTUNE = tf.data.AUTOTUNE

    train_portrait = train_portrait.cache().map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
        buffer_size, seed=123).batch(batch_size)

    train_naruto = train_naruto.cache().map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
        buffer_size, seed=123).batch(batch_size)

    val_portrait = val_portrait.map(
        preprocess_image_val, num_parallel_calls=AUTOTUNE).cache().batch(val_batch_size)

    val_naruto = val_naruto.map(
        preprocess_image_val, num_parallel_calls=AUTOTUNE).cache().batch(val_batch_size)

    test_portrait = test_portrait.map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache()

    test_naruto = test_naruto.map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache()

    return train_portrait, train_naruto, val_portrait, val_naruto, test_portrait, test_naruto


