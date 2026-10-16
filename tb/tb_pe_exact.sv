`timescale 1ns / 1ps

`ifndef PE_EXACT_INCLUDED
`define PE_EXACT_INCLUDED
`include "../rtl/pe_exact.sv"
`endif


module tb_pe_exact;

    // =========================================================================
    // Parameters
    // =========================================================================

    parameter integer CLK_PERIOD      = 10;
    parameter integer PIPELINE_STAGES = 2;
    parameter integer MAX_VECTORS     = 200000;

    // =========================================================================
    // DUT Signals
    // =========================================================================

    reg                     clk;
    reg                     rst_n;
    reg                     en;

    reg signed [7:0]        weight;
    reg signed [7:0]        act;
    reg signed [31:0]       acc_in;

    wire signed [31:0]      acc_out;
    wire signed [7:0]       act_out;
    wire signed [7:0]       weight_out;

    // =========================================================================
    // Testbench Variables
    // =========================================================================

    integer vector_count;
    integer pass_count;
    integer fail_count;
    integer total_vectors;

    // Input vectors
    reg [7:0]  mem_weight [0:MAX_VECTORS-1];
    reg [7:0]  mem_act    [0:MAX_VECTORS-1];
    reg [31:0] mem_acc_in [0:MAX_VECTORS-1];
    reg [31:0] mem_exp    [0:MAX_VECTORS-1];

    // Expected-value pipeline
    reg signed [31:0] exp_pipe [0:PIPELINE_STAGES-1];
    integer           id_pipe  [0:PIPELINE_STAGES-1];
    reg               vld_pipe [0:PIPELINE_STAGES-1];

    integer i;
    integer s;
    integer fd;
    integer r;

    logic signed [7:0]  w_tmp;
    logic signed [7:0]  a_tmp;
    logic signed [31:0] acc_tmp;
    // logic [31:0] exp_tmp;


    // =========================================================================
    // DUT
    // =========================================================================

    pe_exact #(
        .PIPELINE_STAGES(PIPELINE_STAGES)
    ) u_dut (

        .clk          (clk),
        .rst_n        (rst_n),

        .pe_en        (en),
        .clr_acc      (1'b0),

        // IMPORTANT:
        // Each test vector uses its own acc_in.
        .use_ext_acc  (1'b1),

        .weight_in    (weight),
        .act_in       (act),
        .acc_in       (acc_in),

        .acc_out      (acc_out),
        .act_out      (act_out),
        .weight_out   (weight_out)
    );


    // =========================================================================
    // Clock
    // =========================================================================

    initial begin
        clk = 1'b0;

        forever #(CLK_PERIOD / 2)
            clk = ~clk;
    end


    // =========================================================================
    // VCD Dump
    // =========================================================================

    initial begin
        $dumpfile("sim/tb_pe_exact.vcd");
        $dumpvars(0, tb_pe_exact);
    end


    // =========================================================================
    // Load Vectors and Drive DUT
    // =========================================================================

    initial begin

        // ---------------------------------------------------------------------
        // Initial values
        // ---------------------------------------------------------------------

        rst_n         = 1'b0;
        en            = 1'b0;

        weight        = 8'sd0;
        act           = 8'sd0;
        acc_in        = 32'sd0;

        vector_count  = 0;
        pass_count    = 0;
        fail_count    = 0;
        total_vectors = 0;


        // ---------------------------------------------------------------------
        // Open vector file
        // ---------------------------------------------------------------------

        // fd = $fopen("vectors.hex", "r");

        // if (fd == 0) begin
        //     $display("[ERROR] Could not open vectors.hex");
        //     $finish;
        // end


        // // ---------------------------------------------------------------------
        // // Read vectors
        // //
        // // Expected format:
        // //
        // // weight activation acc_in expected
        // //
        // // Example:
        // // FF 02 00000010 0000000E
        // // ---------------------------------------------------------------------

        // while (!$feof(fd) &&
        //        total_vectors < MAX_VECTORS) begin

        //     r = $fscanf(
        //             fd,
        //             "%h %h %h %h\n",
        //             w_tmp,
        //             a_tmp,
        //             acc_tmp,
        //             exp_tmp
        //         );

        //     if (r == 4) begin

        //         mem_weight[total_vectors] = w_tmp;
        //         mem_act[total_vectors]    = a_tmp;
        //         mem_acc_in[total_vectors] = acc_tmp;
        //         mem_exp[total_vectors]    = exp_tmp;

        //         total_vectors = total_vectors + 1;

        //     end

        // end

        // $fclose(fd);

        // $display(
        //     "[INFO] Successfully loaded %0d vectors",
        //     total_vectors
        // );

        
        //constrained randomization of input vectors
        for (i = 0; i < MAX_VECTORS; i = i + 1) begin
        // Generate signed 8-bit values from -128 to 127
        w_tmp   = $urandom();
        a_tmp   = $urandom();
        
        // Generate 32-bit random pattern
        acc_tmp = $urandom();

        mem_weight[i] = w_tmp;
        mem_act[i]    = a_tmp;
        mem_acc_in[i] = acc_tmp;
        mem_exp[i]    = acc_tmp + (w_tmp * a_tmp);
        total_vectors = total_vectors + 1;
    end

        // ---------------------------------------------------------------------
        // Hold reset for 3 clock cycles
        // ---------------------------------------------------------------------

        #(CLK_PERIOD * 3);

        @(negedge clk);

        rst_n = 1'b1;
      //  en    = 1'b1;


        // ---------------------------------------------------------------------
        // Drive vectors
        //
        // Inputs are changed on negedge.
        // DUT samples them on the following posedge.
        // ---------------------------------------------------------------------

        for (i = 0; i < total_vectors; i = i + 1) begin

            @(negedge clk);
            if(i == 0) begin
                en = 1'b1;
            end
            weight = mem_weight[i];
            act    = mem_act[i];
            acc_in = mem_acc_in[i];

        end


        // ---------------------------------------------------------------------
        // Flush pipeline
        // ---------------------------------------------------------------------

        for (i = 0; i < PIPELINE_STAGES + 2; i = i + 1) begin

            @(negedge clk);
            //en = 1'b0;
            weight = 8'sd0;
            act    = 8'sd0;
            acc_in = 32'sd0;

        end


        // Stop driving new transactions
        en = 1'b0;


        // Allow final result to be checked
        #(CLK_PERIOD * 2);


        // ---------------------------------------------------------------------
        // Summary
        // ---------------------------------------------------------------------

        $display("");
        $display("==================================================");
        $display("              TESTBENCH SUMMARY");
        $display("==================================================");

        $display(
            "Total Vectors Tested : %0d",
            vector_count
        );

        $display(
            "Passed               : %0d",
            pass_count
        );

        $display(
            "Failed               : %0d",
            fail_count
        );

        $display("==================================================");


        if ((fail_count == 0) &&
            (vector_count == total_vectors)) begin

            $display("[STATUS] ALL VECTORS PASSED");

        end
        else begin

            $display("[STATUS] VERIFICATION FAILED");

        end

        $display("");

        $finish;

    end

    // ============================================================================
    // Scoreboard Pipeline
    // ============================================================================

    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            for (s = 0; s < PIPELINE_STAGES; s = s + 1) begin
                exp_pipe[s]  <= 32'sd0;
                id_pipe[s]   <= 0;
                vld_pipe[s]  <= 1'b0;
            end

            vector_count <= 0;

        end
        else begin

            // ------------------------------------------------------------
            // Insert expected transaction
            // ------------------------------------------------------------

            if (en && (vector_count < total_vectors)) begin

                exp_pipe[0] <= $signed(mem_exp[vector_count]);
                id_pipe[0]  <= vector_count;
                vld_pipe[0] <= 1'b1;

                vector_count <= vector_count + 1;

            end
            else begin

                vld_pipe[0] <= 1'b0;

            end


            // ------------------------------------------------------------
            // Shift expected pipeline
            // ------------------------------------------------------------

            for (s = 1; s < PIPELINE_STAGES; s = s + 1) begin

                exp_pipe[s] <= exp_pipe[s-1];
                id_pipe[s]  <= id_pipe[s-1];
                vld_pipe[s] <= vld_pipe[s-1];

            end

        end

    end


    // ============================================================================
    // Output Checker
    //
    // IMPORTANT:
    // Check on negedge so that acc_out has already been updated by
    // the DUT's posedge nonblocking assignment.
    // ============================================================================

    always @(negedge clk) begin

        if (rst_n && vld_pipe[PIPELINE_STAGES-1]) begin

            if (acc_out !== exp_pipe[PIPELINE_STAGES-1]) begin

                // $display(
                //     "[FAIL] Vec %0d | Time: %0t | Expected: %08h | Got: %08h",
                //     id_pipe[PIPELINE_STAGES-1 ],
                //     $time,
                //     exp_pipe[PIPELINE_STAGES-1  ],
                //     acc_out
                // );

                fail_count = fail_count + 1;

            end
            else begin

                // $display(
                //     "[PASS] Vec %0d | Time: %0t | Expected: %08h | Got: %08h",
                //     id_pipe[PIPELINE_STAGES-1 ],
                //     $time,
                //     exp_pipe[PIPELINE_STAGES-1  ],
                //     acc_out
                // );

                pass_count = pass_count + 1;

            end

        end

    end

endmodule