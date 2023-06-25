import tensorflow as tf
from tensorflow import keras
import pennylane as qml

class QLSTMCell(keras.Model):
    def __init__(self, input_size, hidden_size, n_qubits, n_layers=1, backend="default.qubit"):
        super(QLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.backend = backend

        self.wires_forget = [f"wire_forget_{i}" for i in range(self.n_qubits)]
        self.wires_inputs = [f"wire_inputs_{i}" for i in range(self.n_qubits)]
        self.wires_update = [f"wire_update_{i}" for i in range(self.n_qubits)]
        self.wires_output = [f"wire_output_{i}" for i in range(self.n_qubits)]

        self.dev_forget = qml.device(self.backend, wires=self.wires_forget)
        self.dev_inputs = qml.device(self.backend, wires=self.wires_inputs)
        self.dev_update = qml.device(self.backend, wires=self.wires_update)
        self.dev_output = qml.device(self.backend, wires=self.wires_output)

        def _circuit_forget(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_forget)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_forget)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_forget]
        self.qlayer_forget = qml.QNode(_circuit_forget, self.dev_forget, interface="tf")

        def _circuit_input(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_inputs)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_inputs, rotation=qml.RY)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_inputs]
        self.qlayer_inputs = qml.QNode(_circuit_input, self.dev_inputs, interface="tf")

        def _circuit_update(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_update)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_update)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_update]
        self.qlayer_update = qml.QNode(_circuit_update, self.dev_update, interface="tf")

        def _circuit_output(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_output)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_output)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_output]
        self.qlayer_output = qml.QNode(_circuit_output, self.dev_output, interface="tf")

        weight_shapes = {"weights": (n_layers, n_qubits)}

        self.cell = keras.layers.Dense(input_size + hidden_size, use_bias=True)
        # default args = xavier_uniform for weight and zeros for bias

        self.VQC = {
            'forget': qml.qnn.KerasLayer(self.qlayer_forget, weight_shapes, n_qubits),
            'inputs': qml.qnn.KerasLayer(self.qlayer_inputs, weight_shapes, n_qubits),
            'update': qml.qnn.KerasLayer(self.qlayer_update, weight_shapes, n_qubits),
            'output': qml.qnn.KerasLayer(self.qlayer_output, weight_shapes, n_qubits)
        }

        self.clayer_out = keras.layers.Dense(n_qubits, use_bias=False)

    def call(self, x, hidden):
        hx, cx = hidden
        gates = tf.concat([x, hx], axis=1)
        gates = self.cell(gates)

        for layer in range(self.n_layers):
            ingate = keras.activations.sigmoid(self.clayer_out(self.VQC['forget'](gates)))
            forgetgate = keras.activations.sigmoid(self.clayer_out(self.VQC['inputs'](gates)))
            cellgate = keras.activations.tanh(self.clayer_out(self.VQC['update'](gates)))
            outgate = keras.activations.sigmoid(self.clayer_out(self.VQC['forget'](gates)))

            cy = tf.math.multiply(cx, forgetgate) + tf.math.multiply(ingate, cellgate)
            hy = tf.math.multiply(outgate, keras.activations.tanh(cy))

        return (hy, cy)
