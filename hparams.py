# Data parameters
sampling_rate = 22050
num_mcep = 36
frame_period = 5.0
n_frames = 128

# Training parameters
num_iterations = 100000
mini_batch_size = 8
generator_learning_rate = 0.00020
discriminator_learning_rate = 0.00010
lambda_cycle = 10
lambda_identity = 10
lambda_triangle = 5
lambda_backward = 5

seed = 65535
