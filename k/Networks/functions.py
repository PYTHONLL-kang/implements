import tensorflow as tf
import k.Losses.functions as Losses
import k.Optimizers.functions as Optimizers

class Neural_network:
    def __init__(self):
        self.network = []

    def to_object(self, param, class_):
        if type(param) == str:
            return class_(param)
        return param

    def compile(self, loss, optimizer, **kwargs):
        self.loss = self.to_object(loss, Losses.function)
        self.optimizer = self.to_object(optimizer, Optimizers.function)
        self.metrics = self.to_object(kwargs.get("metrics"), Losses.function)

        for layer in self.network:
            if not hasattr(layer, 'set_shape'):
                continue

            if layer.input_shape is None:
                layer.input_shape = output_shape

            layer.set_shape()
            output_shape = layer.output_shape

            if hasattr(layer, 'set_vars'):
                layer.set_vars(self.optimizer)

        self.network_depth = len(self.network)

    def predict(self, data):
        self.calcuration_graph = [data]

        for layer in self.network:
            data = layer.forward(data)
            self.calcuration_graph.append(data)

        return data

    def evaluate(self, x, y, **kwargs):
        graph_dict = {'loss' : [], 'metrics' : []}

        pred = self.predict(x)
        loss = self.loss.forward(y, pred)
        graph_dict['loss'] = loss

        if self.metrics is not None:
            graph_dict['metrics'] = self.metrics.forward(y, pred)

        return graph_dict

    def print_expression(self, value):
        return f'{value:.4f}' if value > 0.0001 else f'{value:.4e}'

    def fit(self, x, y, batch_size, epochs, **kwargs):
        verbose = kwargs.get("verbose", 1)
        validation_data = kwargs.get("validation_data")

        graph_dict = {'loss' : [], 'val_loss' : [], 'metrics' : []}
        print_list = ['' for i in range(5)]

        data_num = x.shape[0]

        if batch_size > data_num:
            batch_size = data_num

        mini_batch_num = data_num//batch_size
        epoch_digits = len(str(epochs))

        for epoch in range(1, epochs+1):
            print_list[0] = f'epoch: {epoch:0{epoch_digits}d}'

            loss_sum = 0
            for batch in range(mini_batch_num):
                index = tf.random.uniform(shape=(batch_size,), minval=0, maxval=data_num, dtype=tf.int32)
                batch_x = x[index, :]
                batch_y = y[index, :]

                batch_p = self.predict(batch_x)

                loss = self.loss.forward(batch_y, batch_p)
                gradients = self.loss.backward(batch_y, batch_p)
                for i in range(self.network_depth):
                    gradients = self.network[-i-1].backward(gradients, self.calcuration_graph[-i-2], self.calcuration_graph[-i-1])

                
                print_list[1] = ' | batch : ' + f'{batch}/{mini_batch_num}'
                print_list[2] = ' | loss : ' + self.print_expression(loss)

                if verbose == 2:
                    print(print_list[0], print_list[1], print_list[2])
                
                loss_sum += loss
            loss = loss_sum / mini_batch_num

            if validation_data is not None:
                validation_loss = self.loss.forward(validation_data[1], self.predict(validation_data[0]))
                graph_dict['val_loss'].append(validation_loss)
                print_list[3] = ' | val_loss : ' + self.print_expression(validation_loss)

            if self.metrics is not None:
                metrics_value = self.metrics.forward(batch_y, batch_p)
                graph_dict['metrics'].append(metrics_value)
                print_list[4] = ' | metrics : ' + self.print_expression(metrics_value)

            if verbose == 1:
                print(print_list[0], print_list[2], print_list[3], print_list[4])

            graph_dict['loss'].append(loss)

        return graph_dict