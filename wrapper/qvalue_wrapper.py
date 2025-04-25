import cocotb_test.simulator
import os

def float_to_fixed(val, q=8):
    return int(val * (1 << q))

def fixed_to_float(val, q=8):
    return val / float(1 << q)

def run_hw_accel(old_val, reward, next_max, lr, df):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sim_dir = os.path.join(base_dir, "../sim")
    hw_file = os.path.join(base_dir, "../hw/qvalue_accel.v")
    build_dir = os.path.join(base_dir, "../build")
    input_file = os.path.abspath(os.path.join(sim_dir, "input_buffer.txt"))
    output_file = os.path.abspath(os.path.join(sim_dir, "output_buffer.txt"))

    with open(input_file, "w") as f:
        f.write(f"{old_val} {reward} {next_max} {lr} {df}")

    cocotb_test.simulator.run(
        python_search=[sim_dir],
        toplevel="qvalue_accel",
        module="test_qvalue_accel",
        verilog_sources=[hw_file],
        parameters={},
        sim_build=build_dir,
        extra_env={
            "INPUT_BUFFER_PATH": input_file,
            "OUTPUT_BUFFER_PATH": output_file,
        }
    )

    with open(output_file, "r") as f:
        return int(f.read().strip())

if __name__ == "__main__":
    result = run_hw_accel(512, 256, 768, 204, 243)
    print(f"Result from hardware: {result}")