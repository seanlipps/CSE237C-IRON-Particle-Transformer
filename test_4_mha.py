import numpy as np
import sys
import os
import subprocess

# We keep these imports for the generated strings, 
# even if unused in the main orchestrator logic directly.
from utils.tiling import tile_matrix
import aie.iron as iron
from aie.utils.config import cxx_header_path

# Helper to get absolute path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_standalone_script(script_path):
    """Runs the generated python script as a separate process."""
    cmd = f"{sys.executable} {script_path}"
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"Execution of {script_path} failed.")

def generate_and_run(kernelname, source_filename, code, inputs, output_shape, output_dtype):
    """
    1. Saves input numpy arrays to temp text files.
    2. Writes the generated python script.
    3. Runs the script.
    4. Loads the output text file back into a numpy array.
    """
    
    # 1. Prepare Input Files
    input_paths = []
    for i, inp in enumerate(inputs):
        inp_path = os.path.join(BASE_DIR, f"temp_{kernelname}_in{i}.txt")
        # Ensure input is 1D for savetxt (flatten), the kernel script handles reshaping if needed
        np.savetxt(inp_path, inp, fmt="%d")
        input_paths.append(inp_path)
    
    output_path = os.path.join(BASE_DIR, f"temp_{kernelname}_out.txt")

    # 2. Inject paths and shapes into the code
    # We add a header to the generated code that defines the main block
    header = f"""
import numpy as np
import os
import sys
import aie.iron as iron
from aie.iron import ExternalFunction, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.utils.config import cxx_header_path

SOURCE_FILE = r"{os.path.join(BASE_DIR, source_filename)}"
INCLUDE_DIR = r"{os.path.join(BASE_DIR, "iron_kernels")}"
"""

    # We construct a main block for the generated file that:
    # - Loads the inputs
    # - Calls the JIT function
    # - Saves the output
    
    # Logic to load inputs inside the generated file
    loaders = ""
    inputs = ""
    for i, path in enumerate(input_paths):
        loaders += f'    input{i}_data = iron.tensor(np.loadtxt(r"{path}", dtype=np.int8).flatten(), dtype=np.int8, device="npu")\n'
        inputs += f'input{i}_data, '

    main_block = f"""
if __name__ == "__main__":
    # Load Inputs
{loaders}

    output_placeholder = iron.zeros({output_shape}, dtype=np.{output_dtype.__name__}, device="npu")
    
    {kernelname}_impl({inputs}output_placeholder)

    np.savetxt(r"{output_path}", np.array(output_placeholder, dtype=np.int8), fmt="%d")
"""

    full_content = header + code + main_block
    
    py_filename = f"gen_{kernelname}.py"
    py_filepath = os.path.join(BASE_DIR, py_filename)
    
    with open(py_filepath, "w") as f:
        f.write(full_content)
        
    # 3. Run the script
    run_standalone_script(py_filepath)
    
    # 4. Read output
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Output file {output_path} was not generated.")
        
    result_flat = np.loadtxt(output_path, dtype=output_dtype)
    return result_flat.reshape(output_shape)


# --------------------------------------------------------------------------
# Wrapper Functions
# --------------------------------------------------------------------------

def make_dense(kernelname, filename):
    # Returns a function that matches the signature (input, output) -> but we only use it to get data
    def wrapper(input0, output_buffer):
        code = f"""
@iron.jit(is_placed=False)
def {kernelname}_impl(input0, output):
    N = input0.shape[0]
    N_out = output.shape[0]
    element_type = output.dtype

    in_ty = np.ndarray[(N,), np.dtype[element_type]]
    out_ty = np.ndarray[(N_out,), np.dtype[element_type]]

    of_x = ObjectFifo(in_ty, name="x")
    of_z = ObjectFifo(out_ty, name="z")

    kernel = ExternalFunction(
        "{kernelname}",
        source_file=SOURCE_FILE,
        arg_types=[in_ty, out_ty],
        include_dirs=[cxx_header_path(), INCLUDE_DIR],
    )

    def core_body(of_x, of_z, kernel):
        elem_x = of_x.acquire(1)
        elem_z = of_z.acquire(1)
        kernel(elem_x, elem_z)
        of_x.release(1)
        of_z.release(1)

    worker = Worker(core_body, fn_args=[of_x.cons(), of_z.prod(), kernel])

    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (a_x, c_z):
        rt.start(worker)
        rt.fill(of_x.prod(), a_x) # Fill with the numpy array passed in
        rt.drain(of_z.cons(), c_z, wait=True) # Drain into the output numpy array

    my_program = Program(iron.get_current_device(), rt)
    return my_program.resolve_program(SequentialPlacer())
"""
        # Execute
        result = generate_and_run(kernelname, filename, code, [input0], output_buffer.shape, output_buffer.dtype)
        # Update the main script's buffer
        output_buffer = result
    return wrapper

def make_score(kernelname, filename):
    def wrapper(input0, input1, output_buffer):
        code = f"""
@iron.jit(is_placed=False)
def {kernelname}_impl(input0, input1, output):
    N = input0.shape[0]
    N1 = input1.shape[0]
    N_out = output.shape[0]
    element_type = output.dtype

    in_tx = np.ndarray[(N,), np.dtype[element_type]]
    in_ty = np.ndarray[(N1,), np.dtype[element_type]]
    out_ty = np.ndarray[(N_out,), np.dtype[element_type]]

    of_x = ObjectFifo(in_tx, depth=1, name="x")
    of_y = ObjectFifo(in_ty, depth=1, name="y")
    of_z = ObjectFifo(out_ty, depth=1, name="z")

    kernel = ExternalFunction(
        "{kernelname}",
        source_file=SOURCE_FILE,
        arg_types=[in_tx, in_ty, out_ty],
        include_dirs=[cxx_header_path(), INCLUDE_DIR],
    )

    def core_body(of_x, of_y, of_z, kernel):
        elem_x = of_x.acquire(1)
        elem_y = of_y.acquire(1)
        elem_z = of_z.acquire(1)
        kernel(elem_x, elem_y, elem_z)
        of_x.release(1)
        of_y.release(1)
        of_z.release(1)

    worker = Worker(core_body, fn_args=[of_x.cons(), of_y.cons(), of_z.prod(), kernel])

    rt = Runtime()
    with rt.sequence(in_tx, in_ty, out_ty) as (a_x, a_y, c_z):
        rt.start(worker)
        rt.fill(of_x.prod(), a_x)
        rt.fill(of_y.prod(), a_y)
        rt.drain(of_z.cons(), c_z, wait=True)

    my_program = Program(iron.get_current_device(), rt)
    return my_program.resolve_program(SequentialPlacer())
"""
        result = generate_and_run(kernelname, filename, code, [input0, input1], output_buffer.shape, output_buffer.dtype)
        output_buffer = result
    return wrapper

def make_context(kernelname, filename):
    def wrapper(input0, input1, output_buffer):
        code = f"""
@iron.jit(is_placed=False)
def {kernelname}_impl(input0, input1, output):
    N = input0.shape[0]
    N1 = input1.shape[0]
    N_out = output.shape[0]
    element_type = output.dtype

    in_tx = np.ndarray[(N,), np.dtype[element_type]]
    in_ty = np.ndarray[(N1,), np.dtype[element_type]]
    out_ty = np.ndarray[(N_out,), np.dtype[element_type]]

    of_x = ObjectFifo(in_tx, name="x")
    of_y = ObjectFifo(in_ty, name="y")
    of_z = ObjectFifo(out_ty, name="z")

    kernel = ExternalFunction(
        "{kernelname}",
        source_file=SOURCE_FILE,
        arg_types=[in_tx, in_ty, out_ty],
        include_dirs=[cxx_header_path(), INCLUDE_DIR],
    )

    def core_body(of_x, of_y, of_z, kernel):
        elem_x = of_x.acquire(1)
        elem_y = of_y.acquire(1)
        elem_z = of_z.acquire(1)
        kernel(elem_x, elem_y, elem_z)
        of_x.release(1)
        of_y.release(1)
        of_z.release(1)

    worker = Worker(core_body, fn_args=[of_x.cons(), of_y.cons(), of_z.prod(), kernel])

    rt = Runtime()
    with rt.sequence(in_tx, in_ty, out_ty) as (a_x, a_y, c_z):
        rt.start(worker)
        rt.fill(of_x.prod(), a_x)
        rt.fill(of_y.prod(), a_y)
        rt.drain(of_z.cons(), c_z, wait=True)

    my_program = Program(iron.get_current_device(), rt)
    return my_program.resolve_program(SequentialPlacer())
"""
        result = generate_and_run(kernelname, filename, code, [input0, input1], output_buffer.shape, output_buffer.dtype)
        output_buffer = result
    return wrapper

def make_concat(kernelname, filename):
    def wrapper(input0, input1, output_buffer):
        code = f"""
@iron.jit(is_placed=False)
def {kernelname}_impl(input0, input1, output):
    N = input0.shape[0]
    N1 = input1.shape[0]
    N_out = output.shape[0]
    element_type = output.dtype

    in_tx = np.ndarray[(N,), np.dtype[element_type]]
    in_ty = np.ndarray[(N1,), np.dtype[element_type]]
    out_ty = np.ndarray[(N_out,), np.dtype[element_type]]

    of_x = ObjectFifo(in_tx, name="x")
    of_y = ObjectFifo(in_ty, name="y")
    of_z = ObjectFifo(out_ty, name="z")

    kernel = ExternalFunction(
        "{kernelname}",
        source_file=SOURCE_FILE,
        arg_types=[in_tx, in_ty, out_ty],
        include_dirs=[cxx_header_path(), INCLUDE_DIR],
    )

    def core_body(of_x, of_y, of_z, kernel):
        elem_x = of_x.acquire(1)
        elem_y = of_y.acquire(1)
        elem_z = of_z.acquire(1)
        kernel(elem_x, elem_y, elem_z)
        of_x.release(1)
        of_y.release(1)
        of_z.release(1)

    worker = Worker(core_body, fn_args=[of_x.cons(), of_y.cons(), of_z.prod(), kernel])

    rt = Runtime()
    with rt.sequence(in_tx, in_ty, out_ty) as (a_x, a_y, c_z):
        rt.start(worker)
        rt.fill(of_x.prod(), a_x)
        rt.fill(of_y.prod(), a_y)
        rt.drain(of_z.cons(), c_z, wait=True)

    my_program = Program(iron.get_current_device(), rt)
    return my_program.resolve_program(SequentialPlacer())
"""
        result = generate_and_run(kernelname, filename, code, [input0, input1], output_buffer.shape, output_buffer.dtype)
        output_buffer = result
    return wrapper

def make_output(kernelname, filename):
    def wrapper(input0, input1, output_buffer):
        # NOTE: original output kernel signature was odd (skipped input0 in ExternalFunction args but used 2 inputs in sequence?)
        # Based on user code: output_ly_kernel arg_types=[in_ty, out_ty]. 
        # But core_body takes of_x and of_z. 
        # And Sequence used in_ty, out_ty. 
        # I will strictly follow the provided logic in the original snippet.
        
        code = f"""
@iron.jit(is_placed=False)
def {kernelname}_impl(input0, input1, output):
    N = input0.shape[0]
    N1 = input1.shape[0]
    N_out = output.shape[0]
    element_type = output.dtype

    in_tx = np.ndarray[(N,), np.dtype[element_type]]
    in_ty = np.ndarray[(N1,), np.dtype[element_type]]
    out_ty = np.ndarray[(N_out,), np.dtype[element_type]]

    of_x = ObjectFifo(in_tx, name="x")
    of_y = ObjectFifo(in_ty, name="y")
    of_z = ObjectFifo(out_ty, name="z")

    # Match original logic exactly
    kernel = ExternalFunction(
        "{kernelname}",
        source_file=SOURCE_FILE,
        arg_types=[in_ty, out_ty], 
        include_dirs=[cxx_header_path(), INCLUDE_DIR],
    )

    def core_body(of_x, of_z, kernel):
        elem_x = of_x.acquire(1)
        elem_z = of_z.acquire(1)
        kernel(elem_x, elem_z)
        of_x.release(1)
        of_z.release(1)

    worker = Worker(core_body, fn_args=[of_x.cons(), of_z.prod(), kernel])

    rt = Runtime()
    # Sequence uses in_ty and out_ty
    with rt.sequence(in_ty, out_ty) as (a_x, c_z):
        rt.start(worker)
        rt.fill(of_x.prod(), a_x)
        rt.drain(of_z.cons(), c_z, wait=True)

    my_program = Program(iron.get_current_device(), rt)
    return my_program.resolve_program(SequentialPlacer())
"""
        # We pass input0 and input1, though logic only seems to use one input flow?
        # The user's original code passed `concat_output0, concat_output1`.
        # I will pass both to the file loader so the signature matches.
        result = generate_and_run(kernelname, filename, code, [input0, input1], output_buffer.shape, output_buffer.dtype)
        output_buffer = result
    return wrapper


def main():
    element_type = np.int8
    
    # Load Initial Data
    inp = np.loadtxt("./iron_kernels/test_data/test_4_a0_golden.txt", dtype=np.int8).flatten()
    ref = np.loadtxt("./iron_kernels/test_data/test_4_a1_golden.txt", dtype=np.int8).flatten()

    INPUT_ROWS = 40
    ff_dim = 64
    num_heads = 4
    head_dim = int(ff_dim/num_heads)

    # define kernel I/O
    inp_tensor = iron.tensor(inp, dtype=np.int8, device="npu")
    q_output0 = iron.zeros(INPUT_ROWS*head_dim, dtype=element_type, device="npu")
    q_output1 = iron.zeros(INPUT_ROWS*head_dim, dtype=element_type, device="npu")
    q_output2 = iron.zeros(INPUT_ROWS*head_dim, dtype=element_type, device="npu")
    q_output3 = iron.zeros(INPUT_ROWS*head_dim, dtype=element_type, device="npu")
    k_output0 = iron.zeros(INPUT_ROWS*head_dim, dtype=element_type, device="npu") #40x64 -> 40x64
    k_output1 = iron.zeros(INPUT_ROWS*head_dim, dtype=element_type, device="npu") #40x64 -> 40x64
    k_output2 = iron.zeros(INPUT_ROWS*head_dim, dtype=element_type, device="npu") #40x64 -> 40x64
    k_output3 = iron.zeros(INPUT_ROWS*head_dim, dtype=element_type, device="npu") #40x64 -> 40x64
    v_output0 = iron.zeros(INPUT_ROWS*head_dim, dtype=element_type, device="npu") #40x64 -> 40x64
    v_output1 = iron.zeros(INPUT_ROWS*head_dim, dtype=element_type, device="npu") #40x64 -> 40x64
    v_output2 = iron.zeros(INPUT_ROWS*head_dim, dtype=element_type, device="npu") #40x64 -> 40x64
    v_output3 = iron.zeros(INPUT_ROWS*head_dim, dtype=element_type, device="npu") #40x64 -> 40x64
    score_output0 = iron.zeros(INPUT_ROWS*INPUT_ROWS, dtype=element_type, device="npu") #40x64x40 -> 40x40
    score_output1 = iron.zeros(INPUT_ROWS*INPUT_ROWS, dtype=element_type, device="npu") #40x64x40 -> 40x40
    score_output2 = iron.zeros(INPUT_ROWS*INPUT_ROWS, dtype=element_type, device="npu") #40x64x40 -> 40x40
    score_output3 = iron.zeros(INPUT_ROWS*INPUT_ROWS, dtype=element_type, device="npu") #40x64x40 -> 40x40
    context_output0 = iron.zeros(INPUT_ROWS*head_dim, dtype=element_type, device="npu") #40x40x64 -> 40x64
    context_output1 = iron.zeros(INPUT_ROWS*head_dim, dtype=element_type, device="npu") #40x40x64 -> 40x64
    context_output2 = iron.zeros(INPUT_ROWS*head_dim, dtype=element_type, device="npu") #40x40x64 -> 40x64
    context_output3 = iron.zeros(INPUT_ROWS*head_dim, dtype=element_type, device="npu") #40x40x64 -> 40x64
    concat_output0 = iron.zeros(INPUT_ROWS*head_dim*2, dtype=element_type, device="npu") #40x40x64 -> 40x64
    concat_output1 = iron.zeros(INPUT_ROWS*head_dim*2, dtype=element_type, device="npu") #40x40x64 -> 40x64
    output_output = iron.zeros(INPUT_ROWS*ff_dim, dtype=element_type, device="npu") #40x64 -> 40x64

    # Run Kernels
    make_dense("q1_head0", "iron_kernels/test_4_layer_1_q_head0.cc")(inp_tensor, q_output0)
    make_dense("q1_head1", "iron_kernels/test_4_layer_1_q_head1.cc")(inp_tensor, q_output1)
    make_dense("q1_head2", "iron_kernels/test_4_layer_1_q_head2.cc")(inp_tensor, q_output2)
    make_dense("q1_head3", "iron_kernels/test_4_layer_1_q_head3.cc")(inp_tensor, q_output3)
    
    make_dense("k1_head0", "iron_kernels/test_4_layer_1_k_head0.cc")(inp_tensor, k_output0)
    make_dense("k1_head1", "iron_kernels/test_4_layer_1_k_head1.cc")(inp_tensor, k_output1)
    make_dense("k1_head2", "iron_kernels/test_4_layer_1_k_head2.cc")(inp_tensor, k_output2)
    make_dense("k1_head3", "iron_kernels/test_4_layer_1_k_head3.cc")(inp_tensor, k_output3)
    
    make_dense("v1_head0", "iron_kernels/test_4_layer_1_v_head0.cc")(inp_tensor, v_output0)
    make_dense("v1_head1", "iron_kernels/test_4_layer_1_v_head1.cc")(inp_tensor, v_output1)
    make_dense("v1_head2", "iron_kernels/test_4_layer_1_v_head2.cc")(inp_tensor, v_output2)
    make_dense("v1_head3", "iron_kernels/test_4_layer_1_v_head3.cc")(inp_tensor, v_output3)
    
    make_score("scores1_head0", "iron_kernels/test_4_layer_1_scores_head0.cc")(q_output0, k_output0, score_output0)
    make_score("scores1_head1", "iron_kernels/test_4_layer_1_scores_head1.cc")(q_output1, k_output1, score_output1)
    make_score("scores1_head2", "iron_kernels/test_4_layer_1_scores_head2.cc")(q_output2, k_output2, score_output2)
    make_score("scores1_head3", "iron_kernels/test_4_layer_1_scores_head3.cc")(q_output3, k_output3, score_output3)
    
    make_context("context1_head0", "iron_kernels/test_4_layer_1_context_head0.cc")(score_output0, v_output0, context_output0)
    make_context("context1_head1", "iron_kernels/test_4_layer_1_context_head1.cc")(score_output1, v_output1, context_output1)
    make_context("context1_head2", "iron_kernels/test_4_layer_1_context_head2.cc")(score_output2, v_output2, context_output2)
    make_context("context1_head3", "iron_kernels/test_4_layer_1_context_head3.cc")(score_output3, v_output3, context_output3)
    
    make_concat("concat1_0", "iron_kernels/test_4_layer_1_concat.cc")(context_output0, context_output1, concat_output0)
    make_concat("concat1_1", "iron_kernels/test_4_layer_1_concat.cc")(context_output2, context_output3, concat_output1)
    
    make_output("out1", "iron_kernels/test_4_layer_1_out.cc")(concat_output0, concat_output1, output_output)
    
    # 4. Results & Verification
    print("scores_output0: ", score_output0)
    print("context_output0: ", context_output0)
    print("context_output1: ", context_output1)
    print("concat_output0: ", concat_output0)
    print("concat_output1: ", concat_output1)
    print("output_output: ", output_output)

    # Save debug files
    np.savetxt("q_output0.txt", q_output0, fmt="%d")
    np.savetxt("q_output1.txt", q_output1, fmt="%d")
    np.savetxt("scores_output0.txt", score_output0, fmt="%d")
    np.savetxt("scores_output1.txt", score_output1, fmt="%d")
    np.savetxt("context_output0.txt", context_output0, fmt="%d")
    np.savetxt("context_output1.txt", context_output1, fmt="%d")
    np.savetxt("concat_output0.txt", concat_output0, fmt="%d")
    np.savetxt("concat_output1.txt", concat_output1, fmt="%d")
    np.savetxt("output_output.txt", output_output, fmt="%d")
    
    out_np = output_output

    errors = 0
    for i, (a, r) in enumerate(zip(out_np, ref)):
        if a != r:
            errors += 1

    if errors == 0:
        print("\nPASS!\n")
        sys.exit(0)
    else:
        print(f"\nError count: {errors}")
        print("failed.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()