"""
AIE Model class for building and executing neural network models on AIE.

The AIEModel manages the complete workflow:
1. Build DAG of layers
2. Compute golden reference (NumPy)
3. Generate AIE C++ code
4. Compile and simulate
5. Validate results
"""

import numpy as np
import os
import glob
import shutil
import subprocess
from typing import List, Optional, Tuple, Any, Dict
from utils.tiling import tile_matrix


class AIEModel:
    """
    Main model class for AIE neural network execution.
    """

    def __init__(self, m: int = 4, k: int = 8, n: int = 8, iterations: int = 1):
        self.m = m
        self.k = k
        self.n = n
        self.iterations = iterations

        self.layers: List[Any] = []  # Ordered list of layers
        self.input_map: Dict[Any, List[Tuple[Any, int]]] = {}  # layer -> [(src_layer, port), ...]

        self.input_data: Optional[np.ndarray] = None

    def add_layer(self, layer, inputs: List[Any]):
        """
        Add a layer to the model.

        Automatically assigns layer index and tiling parameters.

        Args:
            layer: Layer object (DenseLayer, MHALayer, etc.)
            inputs: List of source layers (required)
                   - [None]: connect to AIE_IN
                   - [layer1]: single input
                   - [layer1, layer2]: multiple inputs (e.g., ResAdd)
                   Port index is automatically set to the last kernel in each source layer.

        Returns:
            The layer object (for chaining)
        """
        # Assign index
        layer.idx = len(self.layers)

        # Set tiling parameters
        layer.m = self.m
        layer.k = self.k
        layer.n = self.n

        self.layers.append(layer)

        if inputs is None or not isinstance(inputs, list) or len(inputs) == 0:
            raise ValueError("inputs must be a non-empty list. Use [None] to connect to AIE_IN.")

        input_tuples = []
        for src in inputs:
            if src is None:
                input_tuples.append((None, 0))
            else:
                port_idx = src.num_kernels() - 1
                input_tuples.append((src, port_idx))
        self.input_map[layer] = input_tuples

        return layer

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Execute the complete AIE workflow.

        Steps:
        1. Compute golden reference (NumPy) & Save to files
        2. Generate AIE C++ kernel code
        3. Run on NPU using Iron

        Args:
            input_data: Input array (will be saved and tiled)

        Returns:
            Final layer output from golden computation
        """
        self.input_data = input_data

        print("=" * 60)
        print("AIE Model Forward Pass")
        print("=" * 60)

        print("\n[1/3] Computing golden reference...")
        self._compute_golden()

        print("\n[2/3] Generating AIE kernel code...")
        self._generate_kernels()

        print("\n[3/3] Running on NPU using Iron...")
        self._run_simulation()

        return self.layers[-1].outputs['a']

    def _get_layer_inputs(self, layer) -> List[np.ndarray]:
        """
        Get input data for a layer from connected sources.

        Args:
            layer: Layer to get inputs for

        Returns:
            List of input arrays
        """
        inputs = self.input_map.get(layer, [])
        input_data = []

        for src_layer, port_idx in inputs:
            if src_layer is None:
                # Input from AIE_IN
                input_data.append(self.input_data)
            else:
                # Input from another layer's output
                input_data.append(src_layer.outputs['a'])

        return input_data

    def _compute_golden(self):
        """
        Compute golden reference for all layers using NumPy.

        Also saves tiled golden outputs to data/ directory for validation.
        """
        if os.path.exists('data'):
            shutil.rmtree('data')
        os.makedirs('data')

        for i, layer in enumerate(self.layers):
            inputs = self._get_layer_inputs(layer)

            print(f"  Layer {i} ({layer.name}): ", end='')
            output = []
            if "mha" in layer.name:
                print(layer.name)
                output = layer._compute_golden(inputs, i)
            else:
                output = layer._compute_golden(inputs)
            print(f"output shape = {output.shape}")

            output_tiled = tile_matrix(output, self.m, self.n)
            tiled_repeated = np.tile(output_tiled, (self.iterations, 1))
            np.savetxt(f"data/a{i}_golden.txt",
                      tiled_repeated.reshape(-1, 16),
                      fmt="%s", delimiter=" ")

        final_output = self.layers[-1].outputs['a']
        final_tiled = tile_matrix(final_output, self.m, self.n)
        final_repeated = np.tile(final_tiled, (self.iterations, 1))
        np.savetxt("data/out_ref.txt",
                  final_repeated.reshape(-1, 16),
                  fmt="%s", delimiter=" ")

        input_tiled = tile_matrix(self.input_data, self.m, self.k)
        input_repeated = np.tile(input_tiled, (self.iterations, 1))
        np.savetxt("data/input.txt",
                  input_repeated.reshape(-1, 16),
                  fmt="%s", delimiter=" ")

        print(f"  ✓ Golden computation complete. Saved to data/")

    def _generate_kernels(self):
        """Generate C++ kernel files (layer_X.cc) for all layers."""
        for path in glob.glob("iron_kernels/layer_*.cc"):
            os.remove(path)

        for layer in self.layers:
            with open(f"iron_kernels/layer_{layer.idx}.cc", "w") as f:
                layer.generate_kernel_code(f)

        print(f"  ✓ Generated {len(self.layers)} kernel files")

    def _get_layer_input_ports(self, layer) -> List[str]:
        """
        Get input port names for a layer.
        
        Args:
            layer: Layer to get input ports for

        Returns:
            List of port names (e.g., ['AIE_IN', 'dense_0'])
        """
        inputs = self.input_map.get(layer, [])
        port_names = []

        for src_layer, port_idx in inputs:
            if src_layer is None:
                port_names.append("AIE_IN")
            else:
                port_names.append(src_layer.get_output_port(port_idx))

        return port_names

    def _run_simulation(self):
        """Run kernel codes on NPU using Iron"""
        try:
            subprocess.run(["./run_all_layers.sh"], check=True, cwd=".") # this is where we run the top iron script
            print(f"  ✓ Compilation and simulation complete")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Error during compilation/simulation: {e}")
            raise


    def __repr__(self) -> str:
        """String representation."""
        return f"AIEModel(m={self.m}, k={self.k}, n={self.n}, layers={len(self.layers)})"