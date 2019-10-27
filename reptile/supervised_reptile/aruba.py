import tensorflow as tf
from tensorflow.python.ops import control_flow_ops, math_ops, state_ops
from tensorflow.python.framework import constant_op, ops
from tensorflow.python.training import optimizer


class MahalanobisOptimizer(optimizer.Optimizer):

    def __init__(self, aruba, adaptive=0.0, use_locking=False, name='Mahalanobis'):
        super(MahalanobisOptimizer, self).__init__(use_locking, name)
        self._aruba = aruba
        self._adaptive = adaptive

    def _prepare(self):
        if self._adaptive:
            self._coef = ops.convert_to_tensor(self._adaptive, name='coef')

    def _apply_dense(self, grad, var):

        if self._adaptive:
            bsq = self._aruba.get_slot(var, 'bsq')
            gsq = self._aruba.get_slot(var, 'gsq')
            dtype = var.dtype.base_dtype
            coef = math_ops.cast(self._coef, dtype)

            update_var = state_ops.assign_sub(var, tf.sqrt(tf.divide(bsq, gsq)) * grad)
            with tf.control_dependencies([update_var]):
                accumulate_gsq = gsq.assign(gsq + coef * tf.square(grad))
            return control_flow_ops.group(accumulate_gsq)

        eta = self._aruba.get_slot(var, 'eta')
        update_var = state_ops.assign_sub(var, eta*grad)
        return control_flow_ops.group(update_var)


class ARUBAOptimizer(optimizer.Optimizer):

    def __init__(self, eps=1.0, zeta=1.0, p=1.0, decay=0.0, niters=5, learning_rate=None, use_locking=False, train_with_sgd=False, name='ARUBA'):

        assert niters > 1, "ARUBA requires more than one inner iteration"
        super(ARUBAOptimizer, self).__init__(use_locking, name)
        self._eps = eps if learning_rate is None else learning_rate * zeta
        self._zeta = zeta
        self._p = p
        self._niters = niters
        self._decay = decay
        self._twsgd = train_with_sgd

    def _prepare(self):
        self._epssq = ops.convert_to_tensor(self._eps ** 2, name='epssq')
        self._zetasq = ops.convert_to_tensor(self._zeta ** 2, name='zetasq')
        self._power = ops.convert_to_tensor(self._p, name='power')
        self._n = ops.convert_to_tensor(self._niters, name='n')
        if self._decay:
            self._beta = ops.convert_to_tensor(1.0-self._decay, name='beta')
        if self._twsgd:
            self._eta_train = ops.convert_to_tensor(self._eps / self._zeta, name='eta_train')

    def _get_non_slot_var(self, name):
        graph = None if tf.contrib.eager.in_eager_mode() else tf.get_default_graph()
        return self._get_non_slot_variable(name, graph=graph)

    def _constant_slot(self, var, val, slot_name, op_name):
        self._get_or_make_slot(var, constant_op.constant(val, shape=var.get_shape(), dtype=var.dtype.base_dtype), slot_name, op_name)

    def _create_slots(self, var_list):

        first = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(initial_value=0, name='iter', colocate_with=first)
        self._create_non_slot_variable(initial_value=1, name='task', colocate_with=first)
        for v in var_list:
            self._constant_slot(v, self._eps ** 2, 'bsq', self._name)
            self._constant_slot(v, self._zeta ** 2, 'gsq', self._name)
            self._constant_slot(v, self._eps / self._zeta, 'eta', self._name)
            self._zeros_slot(v, 'phi', self._name)

    def _start_task(self, grad, var):

        gsq = self.get_slot(var, 'gsq')
        if self._twsgd:
            eta = math_ops.cast(self._eta_train, var.dtype.base_dtype)
        else:
            eta = self.get_slot(var, 'eta')
        phi = self.get_slot(var, 'phi')

        update_phi = phi.assign(var)
        with tf.control_dependencies([update_phi]):
            update_var = state_ops.assign_sub(var, eta*grad)
            accumulate_gsq = gsq.assign(gsq + tf.square(grad))
        return control_flow_ops.group(update_var, accumulate_gsq)

    def _inner_iter(self, grad, var):

        gsq = self.get_slot(var, 'gsq')
        if self._twsgd:
            eta = math_ops.cast(self._eta_train, var.dtype.base_dtype)
        else:
            eta = self.get_slot(var, 'eta')

        update_var = state_ops.assign_sub(var, eta*grad)
        accumulate_gsq = gsq.assign(gsq + tf.square(grad))
        return control_flow_ops.group(update_var, accumulate_gsq)

    def _end_task(self, grad, var):

        bsq = self.get_slot(var, 'bsq')
        gsq = self.get_slot(var, 'gsq')
        eta = self.get_slot(var, 'eta')
        phi = self.get_slot(var, 'phi')

        dtype = var.dtype.base_dtype
        epssq = math_ops.cast(self._epssq, dtype)
        zetasq = math_ops.cast(self._zetasq, dtype)
        power = math_ops.cast(self._power, dtype)
        task = math_ops.cast(self._get_non_slot_var('task'), dtype)

        if self._twsgd:
            eta_train = math_ops.cast(self._eta_train, dtype)
            update_var = state_ops.assign_sub(var, eta_train*grad)
        else:
            update_var = state_ops.assign_sub(var, eta*grad)
        if self._decay:
            beta = math_ops.cast(self._beta, dtype)
            accumulate_bsq = bsq.assign(beta*bsq + 0.5*tf.square(update_var-phi) + tf.divide(epssq, tf.pow(task+1, power)))
            accumulate_gsq = gsq.assign(beta*gsq + tf.square(grad) + tf.divide(zetasq, tf.pow(task+1, power)))
        else:
            accumulate_bsq = bsq.assign(bsq + 0.5*tf.square(update_var-phi) + tf.divide(epssq, tf.pow(task+1, power)))
            accumulate_gsq = gsq.assign(gsq + tf.square(grad) + tf.divide(zetasq, tf.pow(task+1, power)))
        update_eta = eta.assign(tf.sqrt(tf.divide(accumulate_bsq, accumulate_gsq)))
        return control_flow_ops.group(update_eta)

    def _apply_dense(self, grad, var):
        i = self._get_non_slot_var('iter')
        return tf.case([(tf.equal(i, 0), lambda: self._start_task(grad, var)),
                        (tf.equal(i, self._niters-1), lambda: self._end_task(grad, var))],
                        default=lambda: self._inner_iter(grad, var))

    def _finish(self, update_ops, name_scope):
        i, t = self._get_non_slot_var('iter'), self._get_non_slot_var('task')
        with tf.control_dependencies(update_ops):
            update_iter = i.assign(tf.mod(i + 1, self._n), use_locking=self._use_locking)
            update_task = t.assign(t + tf.cast(tf.equal(update_iter, 0), tf.int32), use_locking=self._use_locking)
        return tf.group(*update_ops + [update_task], name=name_scope)
