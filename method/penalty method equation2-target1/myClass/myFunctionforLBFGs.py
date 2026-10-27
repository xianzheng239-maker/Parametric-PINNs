# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 22:56:43 2024

@author: lenovo
"""
import numpy
import tensorflow as tf
import tensorflow_probability as tfp

class Function_LBFGS():
    def __init__(self):
        super().__init__()
        
        temp = 0.0
    
    def train_lbfgs(self,model,loss_fun,inps,num_inter):
        
        func = self.function_factory(model, loss_fun, inps)
        # convert initial model parameters to a 1D tf.Tensor
        init_params = tf.dynamic_stitch(func.idx, model.trainable_variables)

        # train the model with L-BFGS solver
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=func, initial_position=init_params, max_iterations=num_inter,x_tolerance=0.000001)

        # after training, the final optimized parameters are still in results.position
        # so we have to manually put them back to the model
        func.assign_new_model_parameters(results.position)
        
        return func
    
    def function_factory(self,model, loss, train_x):
        """A factory to create a function required by tfp.optimizer.lbfgs_minimize.

        Args:
            model [in]: an instance of `tf.keras.Model` or its subclasses.
            loss [in]: a function with signature loss_value = loss(pred_y, true_y).
            train_x [in]: the input part of training data.
            train_y [in]: the output part of training data.

        Returns:
            A function that has a signature of:
                loss_value, gradients = f(model_parameters).
        """

        # obtain the shapes of all trainable parameters in the model
        shapes = tf.shape_n(model.trainable_variables)
        n_tensors = len(shapes)

        # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
        # prepare required information first
        count = 0
        idx = [] # stitch indices
        part = [] # partition indices

        for i, shape in enumerate(shapes):
            n = numpy.product(shape)
            idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
            part.extend([i]*n)
            count += n

        part = tf.constant(part)

        @tf.function
        def assign_new_model_parameters(params_1d):
            """A function updating the model's parameters with a 1D tf.Tensor.

            Args:
                params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
            """

            params = tf.dynamic_partition(params_1d, part, n_tensors)
            for i, (shape, param) in enumerate(zip(shapes, params)):
                model.trainable_variables[i].assign(tf.reshape(param, shape))

        # now create a function that will be returned by this factory
        @tf.function
        def f(params_1d):
            """A function that can be used by tfp.optimizer.lbfgs_minimize.

            This function is created by function_factory.

            Args:
               params_1d [in]: a 1D tf.Tensor.

            Returns:
                A scalar loss and the gradients w.r.t. the `params_1d`.
            """

            # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
            with tf.GradientTape() as tape:
                # update the parameters in the model
                assign_new_model_parameters(params_1d)
                # calculate the loss
                loss_value, lossu,lossf, lossn, lossc = loss(train_x)

            # calculate gradients and convert to 1D tf.Tensor
            grads = tape.gradient(loss_value, model.trainable_variables)
            grads = tf.dynamic_stitch(idx, grads)

            # print out iteration & loss
            f.iter.assign_add(1)
            if(f.iter % 500==0):
                tf.print("Iter:", f.iter )
                tf.print("loss:", loss_value )
                tf.print("lossu:", lossu)
                tf.print("lossf:", lossf)
                tf.print("lossn:", lossn)
                tf.print("lossc:", lossc)
                tf.print("=====================================================")

                # store loss value so we can retrieve later
                tf.py_function(f.loss.append, inp=[loss_value], Tout=[])
                tf.py_function(f.lossu.append, inp=[lossu], Tout=[])
                tf.py_function(f.lossf.append, inp=[lossf], Tout=[])

                tf.py_function(f.lossn.append, inp=[lossn], Tout=[])
                tf.py_function(f.lossc.append, inp=[lossc], Tout=[])

            return loss_value, grads

        # store these information as members so we can use them outside the scope
        f.iter = tf.Variable(0)
        f.idx = idx
        f.part = part
        f.shapes = shapes
        f.assign_new_model_parameters = assign_new_model_parameters
        f.loss = []
        f.lossu = []
        f.lossf = []
        f.lossn = []
        f.lossc = []

        return f
        
    