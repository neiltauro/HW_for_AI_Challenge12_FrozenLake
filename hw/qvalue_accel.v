`timescale 1ns/1ps

module qvalue_accel (
    input  [15:0] old_value,
    input  [15:0] reward,
    input  [15:0] next_max,
    input  [15:0] learning_rate,
    input  [15:0] discount_factor,
    output reg  [15:0] new_value
);

    reg [31:0] part1, part2, temp;

    always @(*) begin
        part1 = (16'h0100 - learning_rate) * old_value;
        part2 = learning_rate * (reward + (discount_factor * next_max >> 8));
        temp = (part1 + part2) >> 8;
        new_value = temp[15:0];
    end

endmodule