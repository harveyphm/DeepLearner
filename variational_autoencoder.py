import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

Bernoulli = tf.contrib.distributions.Bernoulli

def xavier_glorot_init(input_dim, output_dim):
	return tf.random_normal(shape = (input_dim, output_dim), stddev = 1./tf.sqrt(input_dim/2.)) 

class FCLayer(object):
	def __init__(self, input_dim, output_dim, activate_function = tf.nn.relu, bias = True):
		self.W = tf.Variable(xavier_glorot_init([input_dim, output_dim]))
		self.bias  = bias
		if bias:
			self.b = tf.Variable(np.zeros(output_dim).astype(np.float32))
		self.activate_function = activate_function

	def forward(self, X):
		if self.bias:
			z = tf.matmul(X, self.W) + self.b
		else:
			z = tf.matmul(X, self.W)
		return z


class VariationalAutoencoder(object):
	def __init__(self, data_dim, latent_layer_sizes, learning_rate = 1e-3):
		#For variational autoencoder we only need the encoder layer sizes
		#because the hidder layer sizes of decoder will be the same

		self.X = tf.placeholder(dtype = tf.float32, shape = (None, data_dim))

		#Build the encoser 
		self.encoder = []

		input_dim = data_dim

		for output_dim in latent_layer_sizes[:-1]:
			hidden_layer = FCLayer(input_dim, output_dim)
			self.encoder.append(hidden_layer)
			input_dim = output_dim

		latent_dim = latent_layer_sizes[-1]
		latent_layer = FCLayer(input_dim, latent_dim*2, activate_function = lambda x:x)
		#the latent layer will out put the mean and std representing the latent vector
		#That is the reason why the output has the shape of 2*latent dim

		self.encoder.append(latent_layer)

		z = self.X
		for layer in self.encoder:
			z = layer.forward(z)
		self.means = z[:,:latent_dim]
		self.std   = tf.nn.solfplus(z[:,latent_dim:]) + 1e-6

		eps = tf.random_normal(tf.shape(latent_dim),dtype = tf.float32,
			mean = 0., stddev = 1.0, name = 'epsilon')

		self.z = self.mean + self.std*eps

		#build the decoder
		self.decoder = []
		input_dim = latent_dim
		for output_dim in reversed(latent_layer_sizes[:-1]):
			layer = FCLayer(input_dim, output_dim)
			self.decoder.append(layer)
			input_dim = output_dim

		#The decoder's final layer should go to sigmoid
		#to make the final output a binary probability (e.q Bernoulli)
		#However, Bernoulli accepts logits (pre-sigmoid)
		#Therefore, we dont pass the output to an activate function (Sigmoid)
		layer = FCLayer(input_dim, output_dim, activate_function = lambda x:x)
		self.decoder.append(layer)

		z = self.z
		for layer in self.decoder:
			z = layer.forward(z)

		logits = z
		prior_predictive_dist =Bernoulli(logits = logits)
		self.prior_predictive = prior_predictive_dist.sample()
		self.prior_predictive_prob = tf.nn.sigmoid(logits)

		#Build the KL-divergence
		kl = -tf.log(self.std)+ 0.5*(self.std**2 + self.mean**2) -0.5
		kl = tf.reduce_sum(kl, axis = 1)

		#Build the expected log likelihood 
		expected_log_likelihood = tf.reduce_sum(
				self.X_hat_dist_tribution.log_prob(self.X),
				1)


		self.elbo = tf.reduce_sum(expected_log_likelihood - kl)
		self.train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(-self.elbo)

		self.init_op = tf.global_variables_initializer()
		self.sess = tf.InteractiveSession()
		self.sess.run(self.init_op)


	def fit(self, X, epochs = 30, batch_size = 64):
		cost = []
		n_batches = len(X) // batch_size
		print("Number of batch ", n_batches)
		for i in range(epochs):
			print("epoch:", i)
			np.random.shuffle(X)
			for j in range(n_batches):
				batch = X[j*batch_size:(j+1)*batch_size]
				_,c = self.sess.run((self.train_op, self.elbo), 
					feed_dict = {
					self.X : batch
					})
				c /= batch_size 
				cost.append(c)
				if j%100 == 0:
					print("iter " j ", cost ", c)
		plt.plot(cost)
		plt.show()

	def transform(self,X):
		self.sess.run(
			self.means,
			feed_dict = {
			self.X = X
			})

	def prior_predictive_with_input(self, Z):
		return self.sess.run(
			self.pior_predictive_from_input_probs,
			feed_dict = {
			self.Z_input:Z
			})

	def posterior_predictive_sample(self,X):
		return self.sess.run(
			self.posterior_predictive, feed_dict = {self.X : X})

	def prior_predictive_sample_with_probs(self):
		return self.sess.run((self.prior_predictive, self.prior_predictive_prob))

		



