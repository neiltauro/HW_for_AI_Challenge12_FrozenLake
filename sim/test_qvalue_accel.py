import cocotb
from cocotb.triggers import Timer
import os

@cocotb.test()
async def qvalue_test(dut):
    await Timer(1, units="ns")

    input_path = os.environ["INPUT_BUFFER_PATH"]
    output_path = os.environ["OUTPUT_BUFFER_PATH"]

    with open(input_path, "r") as f:
        values = list(map(int, f.read().strip().split()))

    dut.old_value.value = values[0]
    dut.reward.value = values[1]
    dut.next_max.value = values[2]
    dut.learning_rate.value = values[3]
    dut.discount_factor.value = values[4]

    await Timer(1, units="ns")

    with open(output_path, "w") as f:
        f.write(str(int(dut.new_value.value)))