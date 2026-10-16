`timescale 1ns / 1ps

module tb_pe_wrapper;

    localparam integer PIPELINE_STAGES   = 2;
    localparam integer TRUNC_BITS        = 4;
    localparam bit     ROUND_BIAS        = 1;
    localparam real    MAX_TOLERATED_MRE = 5.0; // 5% application budget

    localparam int NUM_FILTERS = 6;
    localparam int ROWS        = 28;
    localparam int COLS        = 28;
    localparam int KERNEL_SIZE = 5;

    logic                    clk;
    logic                    rst_n;
    logic                    pe_en;
    logic                    clr_acc;
    logic                    use_ext_acc;

    logic signed [7:0]       act_in;
    logic signed [7:0]       weight_in;
    logic signed [31:0]      acc_in;

    logic signed [7:0]       act_out;
    logic signed [7:0]       weight_out;
    logic signed [31:0]      acc_out;

    pe_wrapper #(
        .PIPELINE_STAGES (PIPELINE_STAGES),
        .USE_APPROX      (0),
        .TRUNC_BITS      (TRUNC_BITS),
        .ROUND_BIAS      (ROUND_BIAS)
    ) dut (
        .clk             (clk),
        .rst_n           (rst_n),
        .pe_en           (pe_en),
        .clr_acc         (clr_acc),
        .use_ext_acc     (use_ext_acc),
        .act_in          (act_in),
        .weight_in       (weight_in),
        .acc_in          (acc_in),
        .act_out         (act_out),
        .weight_out      (weight_out),
        .acc_out         (acc_out)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    int unsigned total_tests       = 0;
    int unsigned mismatch_count    = 0;
    int unsigned non_zero_tests    = 0;
    int unsigned fatal_errors      = 0;
    real         total_abs_error   = 0.0;
    real         total_rel_error   = 0.0;
    int unsigned max_abs_error     = 0;

    initial begin
        rst_n       = 1'b0;
        pe_en       = 1'b0;
        clr_acc     = 1'b0;
        use_ext_acc = 1'b0;
        act_in      = 8'sd0;
        weight_in   = 8'sd0;
        acc_in      = 32'sd0;

        #20;
        rst_n = 1'b1;
        #10;

`ifdef APPROX_PE
        $display(">> Running Testbench in APPROXIMATE PE Mode (`APPROX_PE)");
`else
        $display(">> Running Testbench in EXACT PE Mode");
`endif

        for (int f = 0; f < NUM_FILTERS; f++) begin
            for (int r = 0; r < ROWS; r++) begin
                for (int c = 0; c < COLS; c++) begin
                    logic signed [31:0] expected_acc;
                    logic signed [31:0] hw_acc;

                    // 1. Synchronous Clear
                    @(posedge clk);
                    clr_acc <= 1'b1;
                    pe_en   <= 1'b0;
                    @(posedge clk);
                    clr_acc <= 1'b0;
                    pe_en   <= 1'b1;

                    expected_acc = 32'sd0;

                    // 2. Stream Kernel Inputs
                    for (int k = 0; k < KERNEL_SIZE; k++) begin
                        logic signed [7:0] stim_act;
                        logic signed [7:0] stim_weight;

                        stim_act    = 8'($signed((r * 7 + c * 11 + k * 13 + f) % 255 - 128));
                        stim_weight = 8'($signed((f * 17 + k * 19) % 255 - 128));

                        act_in       <= stim_act;
                        weight_in    <= stim_weight;
                        expected_acc += stim_act * stim_weight;

                        @(posedge clk);
                    end

                    // 3. Drain Pipeline (Keep pe_en=1 with zero inputs)
                    act_in    <= 8'sd0;
                    weight_in <= 8'sd0;
                    repeat (PIPELINE_STAGES) @(posedge clk);

                    // 4. Sample and Disable
                    hw_acc = acc_out;
                    pe_en  <= 1'b0;

                    // 5. Evaluate Metrics
                    begin
                        int diff;
                        real rel_err;

                        diff = (hw_acc >= expected_acc) ? (hw_acc - expected_acc) 
                                                         : (expected_acc - hw_acc);

                        total_tests++;
                        total_abs_error += diff;

                        if (diff > max_abs_error) begin
                            max_abs_error = diff;
                        end

                        if (diff != 0) begin
                            mismatch_count++;
                        end

                        if (expected_acc != 0) begin
                            rel_err = real'(diff) / real'((expected_acc < 0) ? -expected_acc : expected_acc);
                            total_rel_error += rel_err;
                            non_zero_tests++;
                        end

`ifndef APPROX_PE
                        if (diff != 0) begin
                            $display("ERROR: tb/tb_pe_wrapper.sv: EXACT MISMATCH @ Filter %0d, Row %0d, Col %0d: HW = %0d, Expected = %0d",
                                     f, r, c, hw_acc, expected_acc);
                            fatal_errors++;
                        end
`endif
                    end
                end
            end
        end

        // Report Statistics
        begin
            real final_mre;
            real final_mae;
            real final_er;

            final_mae = total_abs_error / real'(total_tests);
            final_mre = (total_rel_error / real'(non_zero_tests)) * 100.0;
            final_er  = (real'(mismatch_count) / real'(total_tests)) * 100.0;

            $display("\n=======================================================");
            $display("                SIMULATION REPORT                     ");
            $display("=======================================================");
            $display("Total Patterns Tested    : %0d", total_tests);
            $display("Mismatched Patterns      : %0d", mismatch_count);
            $display("Error Rate (ER)          : %0.2f %%", final_er);
            $display("Mean Absolute Error (MAE): %0.3f", final_mae);
            $display("Mean Relative Error (MRE): %0.3f %%", final_mre);
            $display("Max Absolute Error (WCAE): %0d", max_abs_error);
            $display("=======================================================");

`ifndef APPROX_PE
            if (fatal_errors == 0) begin
                $display("STATUS: PASSED (Exact Bit-True Match)\n");
            end else begin
                $display("STATUS: FAILED (%0d mismatches observed)\n", fatal_errors);
            end
`else
            if (final_mre <= MAX_TOLERATED_MRE) begin
                $display("STATUS: PASSED (Achieved %0.2f%% MRE <= %0.2f%% budget)\n", final_mre, MAX_TOLERATED_MRE);
            end else begin
                $display("STATUS: FAILED (MRE %0.2f%% exceeded %0.2f%% budget)\n", final_mre, MAX_TOLERATED_MRE);
            end
`endif
        end

        $finish;
    end

endmodule