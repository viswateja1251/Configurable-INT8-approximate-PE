`timescale 1ns/1ps

module tb_pe_exact;

    // -------------------------------------------------------------------------
    // Parameters & Interface Signals
    // -------------------------------------------------------------------------
    localparam int PIPELINE_STAGES = 2;

    logic        clk;
    logic        rst_n;
    logic        pe_en;
    logic        clr_acc;
    logic        use_ext_acc;

    logic signed [7:0]  act_in;
    logic signed [7:0]  weight_in;
    logic signed [31:0] acc_in;

    logic signed [7:0]  act_out;
    logic signed [7:0]  weight_out;
    logic signed [31:0] acc_out;

    // Memory array matching lenet_int8.hex (10,814 bytes allocated in 16KB)
    logic [7:0] raw_mem [0:16383];

    // Structured multidimensional tensor arrays
    logic signed [7:0] img_in     [0:27][0:27];
    logic signed [7:0] conv1_w    [0:5][0:4][0:4];
    logic signed [7:0] conv1_gold [0:5][0:27][0:27];

    // Tracking Counters
    integer total_verified = 0;
    integer error_count    = 0;

    // -------------------------------------------------------------------------
    // Device Under Test (DUT) Instantiation
    // -------------------------------------------------------------------------
    pe_exact #(
        .PIPELINE_STAGES(PIPELINE_STAGES)
    ) dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .pe_en       (pe_en),
        .clr_acc     (clr_acc),
        .use_ext_acc (use_ext_acc),
        .act_in      (act_in),
        .weight_in   (weight_in),
        .acc_in      (acc_in),
        .act_out     (act_out),
        .weight_out  (weight_out),
        .acc_out     (acc_out)
    );

    // -------------------------------------------------------------------------
    // Clock Generation (100 MHz, 10ns period)
    // -------------------------------------------------------------------------
    initial clk = 0;
    always #5 clk = ~clk;

    // -------------------------------------------------------------------------
    // Task: Read Hex File and Map Tensors
    // -------------------------------------------------------------------------
    task automatic load_and_map_tensors;
        $display("[TB] Loading lenet_int8.hex into testbench memory...");
        $readmemh("lenet_int8.hex", raw_mem);

        // Map Image (Addr: 0x0000 -> 784 bytes)
        for (int r = 0; r < 28; r++) begin
            for (int c = 0; c < 28; c++) begin
                img_in[r][c] = $signed(raw_mem[r * 28 + c]);
            end
        end

        // Map Conv1 Weights (Addr: 0x0310 = 784 -> 150 bytes)
        for (int f = 0; f < 6; f++) begin
            for (int ky = 0; ky < 5; ky++) begin
                for (int kx = 0; kx < 5; kx++) begin
                    conv1_w[f][ky][kx] = $signed(raw_mem[784 + (f * 25 + ky * 5 + kx)]);
                end
            end
        end

        // Map Golden Requantized Reference (Addr: 0x03A6 = 934 -> 4704 bytes)
        for (int f = 0; f < 6; f++) begin
            for (int r = 0; r < 28; r++) begin
                for (int c = 0; c < 28; c++) begin
                    conv1_gold[f][r][c] = $signed(raw_mem[934 + (f * 784 + r * 28 + c)]);
                end
            end
        end
        $display("[TB] Tensor mapping finished successfully.");
    endtask

    // -------------------------------------------------------------------------
    // Function: Zero-Padding (padding=2) Helper
    // -------------------------------------------------------------------------
    function automatic logic signed [7:0] get_padded_pixel(int r, int c);
        if (r < 0 || r >= 28 || c < 0 || c >= 28) begin
            return 8'sd0;
        end else begin
            return img_in[r][c];
        end
    endfunction

    // -------------------------------------------------------------------------
    // Verification Routine
    // -------------------------------------------------------------------------
    initial begin
        // Signal Initialization
        rst_n       = 1'b0;
        pe_en       = 1'b0;
        clr_acc     = 1'b0;
        use_ext_acc = 1'b0; // Use internal accumulator loop
        act_in      = 8'sd0;
        weight_in   = 8'sd0;
        acc_in      = 32'sd0;

        load_and_map_tensors();

        // Apply Reset
        repeat (3) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        $display("[TB] Starting Day 7 Numerical Equivalence Checks across LeNet Conv1...");

        // Loop over 6 filters, 28 rows, and 28 columns (4704 total output pixels)
        for (int f = 0; f < 6; f++) begin
            for (int r = 0; r < 28; r++) begin
                for (int c = 0; c < 28; c++) begin

                    // 1. Calculate Expected INT32 Accumulator value on TB side
                    int expected_acc = 0;
                    for (int ky = 0; ky < 5; ky++) begin
                        for (int kx = 0; kx < 5; kx++) begin
                            int pr = r - 2 + ky;
                            int pc = c - 2 + kx;
                            expected_acc += int'(get_padded_pixel(pr, pc)) * int'(conv1_w[f][ky][kx]);
                        end
                    end

                    // 2. Pulse clr_acc for 1 cycle to clear stage registers & acc_out
                    @(posedge clk);
                    clr_acc <= 1'b1;
                    pe_en   <= 1'b0;
                    act_in  <= 8'sd0;
                    weight_in <= 8'sd0;

                    @(posedge clk);
                    clr_acc <= 1'b0;
                    pe_en   <= 1'b1;

                    // 3. Stream in the 25 MAC operations (5x5 filter patch)
                    for (int ky = 0; ky < 5; ky++) begin
                        for (int kx = 0; kx < 5; kx++) begin
                            int pr = r - 2 + ky;
                            int pc = c - 2 + kx;

                            act_in    <= get_padded_pixel(pr, pc);
                            weight_in <= conv1_w[f][ky][kx];
                            @(posedge clk);
                        end
                    end

                    // 4. Stop feeding new inputs; keep pe_en=1 to drain the pipeline
                    act_in    <= 8'sd0;
                    weight_in <= 8'sd0;

                    // Drain pipeline based on PIPELINE_STAGES parameter
                    repeat (PIPELINE_STAGES) @(posedge clk);
                    pe_en <= 1'b0; // Freeze accumulator

                    // 5. Verify Hardware Accumulator vs Golden Software Accumulator
                    if (acc_out !== expected_acc) begin
                        $error("MISMATCH @ Filter %0d, Row %0d, Col %0d: HW = %0d, Expected = %0d", 
                                f, r, c, acc_out, expected_acc);
                        error_count++;
                    end

                    total_verified++;
                end
            end
        end

        // ---------------------------------------------------------------------
        // Final Status Closure Check
        // ---------------------------------------------------------------------
        $display("\n=======================================================");
        if (error_count == 0) begin
            $display("SUCCESS: Day 7 Milestone 1 Met![cite: 1]");
            $display("100%% Numerical Equivalence verified across all %0d Conv1 tensors.[cite: 1]", total_verified);
            $display("Ready to tag release: v0.1-baseline-exact[cite: 1]");
        end else begin
            $display("FAILURE: %0d mismatches observed out of %0d tests.", error_count, total_verified);
        end
        $display("=======================================================\n");

        $finish;
    end

endmodule