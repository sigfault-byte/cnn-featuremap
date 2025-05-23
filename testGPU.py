# import tensorflow as tf
# import time

# # Check GPU ??
# print("GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# # Test Matrix Multiplication Speed
# A = tf.random.normal([10000, 10000])
# B = tf.random.normal([10000, 10000])

# print("TensorFlow on GPU test:")
# start = time.time()
# C = tf.matmul(A, B)
# end = time.time()

# print(f"GPU TensorFlow test completed in {end - start:.4f} seconds")
import tensorflow as tf
print(dir(tf))
