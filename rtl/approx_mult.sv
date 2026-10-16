`timescale 1ns / 1ps

module approx_mult #(
    parameter integer WIDTH      = 8,
    parameter integer TRUNC_BITS = 4,
    parameter bit     ROUND_BIAS = 1
)(
    input  logic signed [WIDTH-1:0]   a,
    input  logic signed [WIDTH-1:0]   b,
    output logic signed [2*WIDTH-1:0] product
);

    logic               sign_a, sign_b, sign_prod;
    logic [WIDTH-1:0]   mag_a, mag_b;
    logic [2*WIDTH-1:0] unsigned_prod;

    assign sign_a    = a[WIDTH-1];
    assign sign_b    = b[WIDTH-1];
    assign sign_prod = sign_a ^ sign_b;

    // Direct two's-complement negation without truncating typecasts
    assign mag_a = sign_a ? (-a) : a;
    assign mag_b = sign_b ? (-b) : b;

    always_comb begin
        unsigned_prod = '0;

        // Bypass computation if either operand is zero
        if ((mag_a != '0) && (mag_b != '0)) begin
            // Add constant rounding bias at column TRUNC_BITS - 1
            if (ROUND_BIAS && (TRUNC_BITS > 0)) begin
                unsigned_prod = unsigned_prod + (16'd1 << (TRUNC_BITS - 1));
            end

            // Accumulate unpruned partial products using 16-bit shift base
            for (int i = 0; i < WIDTH; i++) begin
                for (int j = 0; j < WIDTH; j++) begin
                    if ((i + j) >= TRUNC_BITS) begin
                        if (mag_a[i] && mag_b[j]) begin
                            unsigned_prod = unsigned_prod + (16'd1 << (i + j));
                        end
                    end
                end
            end
        end
    end

    // Reconstruct signed two's-complement product
    always_comb begin
        if (a == '0 || b == '0) begin
            product = '0;
        end else if (sign_prod) begin
            product = -$signed(unsigned_prod);
        end else begin
            product = $signed(unsigned_prod);
        end
    end

endmodule