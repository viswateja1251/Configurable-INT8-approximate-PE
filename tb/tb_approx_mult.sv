`timescale 1ns/1ps

module tb_approx_mult;

    localparam int WIDTH      = 8;
    localparam int TRUNC_BITS = 4;
    localparam bit ROUND_BIAS = 1;

    logic [WIDTH-1:0]   tb_a;
    logic [WIDTH-1:0]   tb_b;
    logic [2*WIDTH-1:0] approx_out;

    // Instantiate DUT
    approx_mult #(
        .WIDTH(WIDTH),
        .TRUNC_BITS(TRUNC_BITS),
        .ROUND_BIAS(ROUND_BIAS)
    ) dut (
        .a(tb_a),
        .b(tb_b),
        .product(approx_out)
    );

    int unsigned total_cases;
    int unsigned error_cases;
    int unsigned non_zero_cases;
    real         total_rel_error;
    real         total_abs_error;
    int unsigned max_abs_error;

    initial begin
        total_cases     = 0;
        error_cases     = 0;
        non_zero_cases  = 0;
        total_rel_error = 0.0;
        total_abs_error = 0.0;
        max_abs_error   = 0;

        $display("------------------------------------------------------------");
        $display("Starting Exhaustive Sweep: WIDTH=%0d, TRUNC=%0d, ROUND=%0d", 
                 WIDTH, TRUNC_BITS, ROUND_BIAS);
        $display("------------------------------------------------------------");

        for (int i = 0; i < (1 << WIDTH); i++) begin
            for (int j = 0; j < (1 << WIDTH); j++) begin
                tb_a = WIDTH'(i);
                tb_b = WIDTH'(j);
                #1; // Combinational propagation

                begin
                    logic [2*WIDTH-1:0] exact_out;
                    int diff;
                    real rel_err;

                    exact_out = tb_a * tb_b;
                    diff = (exact_out >= approx_out) ? (exact_out - approx_out) 
                                                     : (approx_out - exact_out);

                    total_cases++;
                    total_abs_error += diff;

                    if (diff > max_abs_error) begin
                        max_abs_error = diff;
                    end

                    if (diff != 0) begin
                        error_cases++;
                    end

                    if (exact_out != 0) begin
                        rel_err = real'(diff) / real'(exact_out);
                        total_rel_error += rel_err;
                        non_zero_cases++;
                    end
                end
            end
        end

        // Report Statistics
        $display("\n===================== VERIFICATION RESULTS =====================");
        $display("Total Patterns Evaluated : %0d", total_cases);
        $display("Error Count              : %0d", error_cases);
        $display("Error Rate (ER)          : %0.2f %%", (real'(error_cases) / real'(total_cases)) * 100.0);
        $display("Mean Absolute Error (MAE): %0.3f", total_abs_error / real'(total_cases));
        $display("Mean Relative Error (MRE): %0.3f %%", (total_rel_error / real'(non_zero_cases)) * 100.0);
        $display("Max Absolute Error (WCAE): %0d", max_abs_error);
        $display("=================================================================\n");

        $finish;
    end

endmodule