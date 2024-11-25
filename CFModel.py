# from keras.layers import Embedding, Reshape, Dot, Input
# from keras.models import Model
# import numpy as np

# class CFModel(Model):
#     # The constructor for the class
#     def __init__(self, n_users, m_items, k_factors, **kwargs):
#         super(CFModel, self).__init__(**kwargs)
#         # P is the embedding layer that creates a User by latent factors matrix
#         # If the input is a user_id, P returns the latent factor vector for that user
#         self.P = Embedding(n_users, k_factors, input_length=1)
#         self.P_reshape = Reshape((k_factors,))

#         # Q is the embedding layer that creates a Movie by latent factors matrix
#         # If the input is a movie_id, Q returns the latent factor vector for that movie
#         self.Q = Embedding(m_items, k_factors, input_length=1)
#         self.Q_reshape = Reshape((k_factors,))

#         self.dot_product = Dot(axes=1)

#     # Implementing the call method for the Functional API
#     def call(self, inputs):
#         user_id, item_id = inputs

#         user_embedding = self.P(user_id)
#         user_embedding = self.P_reshape(user_embedding)

#         item_embedding = self.Q(item_id)
#         item_embedding = self.Q_reshape(item_embedding)

#         return self.dot_product([user_embedding, item_embedding])

#     # The rate function to predict user's rating of unrated items
#     def rate(self, user_id, item_id):
#         return self.predict([np.array([user_id]), np.array([item_id])])[0][0]

# # A simple implementation of matrix factorization for collaborative filtering expressed as a Keras Sequential model

# # Keras uses TensorFlow tensor library as the backend system to do the heavy compiling

from keras.layers import Embedding, Reshape, Dot, Input
from keras.models import Model
import numpy as np

class CFModel(Model):
    # The constructor for the class
    def __init__(self, n_users, m_items, k_factors, **kwargs):
        # User embedding
        user_input = Input(shape=(1,), name='user_input')
        P = Embedding(n_users, k_factors, name='user_embedding')(user_input)
        P = Reshape((k_factors,), name='user_reshape')(P)

        # Item embedding
        item_input = Input(shape=(1,), name='item_input')
        Q = Embedding(m_items, k_factors, name='item_embedding')(item_input)
        Q = Reshape((k_factors,), name='item_reshape')(Q)

        # Dot product of user and item embeddings
        R = Dot(axes=1, name='dot_product')([P, Q])

        # Initialize the Model
        super(CFModel, self).__init__(inputs=[user_input, item_input], outputs=R, **kwargs)

    # The rate function to predict user's rating of unrated items
    def rate(self, user_id, item_id):
        return self.predict([np.array([user_id]), np.array([item_id])])[0][0]