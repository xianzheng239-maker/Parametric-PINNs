import numpy
import tensorflow as tf
import tensorflow_probability as tfp


class Function_LBFGS():
    def __init__(self):
        super().__init__()

        temp = 0.0

    def train_lbfgs(self, model, loss_fun, inps, num_inter):
        func = self.function_factory(model, loss_fun, inps)
        # convert initial model parameters to a 1D tf.Tensor
        init_params = tf.dynamic_stitch(func.idx, model.trainable_variables)

        # train the model with L-BFGS solver
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=func,
            initial_position=init_params,
            max_iterations=num_inter,
            x_tolerance=0.000001
        )

        # after training, the final optimized parameters are still in results.position
        # so we have to manually put them back to the model
        func.assign_new_model_parameters(results.position)

        return func

    def function_factory(self, model, loss, train_x):
        """A factory to create a function required by tfp.optimizer.lbfgs_minimize."""

        # obtain the shapes of all trainable parameters in the model
        shapes = [var.shape for var in model.trainable_variables]
        n_tensors = len(shapes)

        # prepare required information for tf.dynamic_stitch and tf.dynamic_partition
        count = 0
        idx = []  # stitch indices
        part = []  # partition indices

        for i, shape in enumerate(shapes):
            # 确保使用整数计算
            n = int(numpy.prod(shape))  # 转换为整数
            idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
            part.extend([i] * n)
            count += n

        part = tf.constant(part)

        @tf.function
        def assign_new_model_parameters(params_1d):
            """Update the model's parameters with a 1D tf.Tensor."""
            params = tf.dynamic_partition(params_1d, part, n_tensors)
            for i, (shape, param) in enumerate(zip(shapes, params)):
                model.trainable_variables[i].assign(tf.reshape(param, shape))

        # Create the function that will be used by L-BFGS
        def f(params_1d):
            """Function that can be used by tfp.optimizer.lbfgs_minimize."""
            # Update the parameters in the model
            assign_new_model_parameters(params_1d)

            # Calculate the loss and gradients
            with tf.GradientTape() as tape:

                tape.watch(model.trainable_variables)
                barrier_term, loss_value, lossu, lossf, lossn, lossc = loss(train_x)

            # Convert all loss components to scalars if they are not
                if hasattr(barrier_term, 'shape') and len(barrier_term.shape) > 0:
                    barrier_term = tf.reduce_mean(barrier_term)

                if hasattr(loss_value, 'shape') and len(loss_value.shape) > 0:
                    loss_value = tf.reduce_mean(loss_value)
                if hasattr(lossu, 'shape') and len(lossu.shape) > 0:
                    lossu = tf.reduce_mean(lossu)

                if hasattr(lossf, 'shape') and len(lossf.shape) > 0:
                    lossf = tf.reduce_mean(lossf)

                if hasattr(lossn, 'shape') and len(lossn.shape) > 0:
                    lossn = tf.reduce_mean(lossn)
                if hasattr(lossc, 'shape') and len(lossc.shape) > 0:
                    lossc = tf.reduce_mean(lossc)

                # Tiny L2 to ensure every parameter participates in the graph (avoid None grads)
                eps = tf.constant(1e-10, dtype=tf.float64)
                l2_reg = eps * tf.add_n([tf.reduce_sum(tf.square(v)) for v in model.trainable_variables])
                loss_value = loss_value + l2_reg

            # Calculate gradients
            grads = tape.gradient(loss_value, model.trainable_variables)

            # Handle None gradients by replacing with zeros
            processed_grads = []
            for grad, var in zip(grads, model.trainable_variables):
                if grad is None:
                    zero_grad = tf.zeros_like(var)
                    processed_grads.append(zero_grad)
                    print(f"Replaced None gradient for variable: {var.name}")
                else:
                    processed_grads.append(grad)

            # Flatten all gradients to 1D and concatenate
            grads_flat = [tf.reshape(g, [-1]) for g in processed_grads]
            grads_concat = tf.concat(grads_flat, axis=0)

            # Ensure the gradient tensor is properly shaped as a 1D vector
            grads_concat = tf.reshape(grads_concat, [-1])

            # Print iteration info
            f.iter.assign_add(1)
            if f.iter % 500 == 0:
                tf.print("Iter:", f.iter)
                tf.print("barrier_term:", barrier_term)
                tf.print("loss:", loss_value)
                tf.print("lossu:", lossu)
                tf.print("lossf:", lossf)
                tf.print("lossn:", lossn)
                tf.print("lossc:", lossc)
                tf.print("=====================================================")

                # Store loss values as Python floats
                f.loss.append(float(loss_value.numpy()))
                f.barrier_term.append(float(barrier_term.numpy()))
                f.lossu.append(float(lossu.numpy()))
                f.lossf.append(float(lossf.numpy()))
                f.lossn.append(float(lossn.numpy()))
                f.lossc.append(float(lossc.numpy()))

            return loss_value, grads_concat

        # Store these as attributes
        f.iter = tf.Variable(0)
        f.idx = idx
        f.part = part
        f.shapes = shapes
        f.assign_new_model_parameters = assign_new_model_parameters
        f.loss = []
        f.barrier_term=[]
        f.lossu = []
        f.lossf_all = []
        f.lossf = []
        f.lossf_add1 = []
        f.lossf_add2 = []
        f.lossn = []
        f.lossc = []

        return f
