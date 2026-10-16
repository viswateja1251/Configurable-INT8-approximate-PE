// ============================================================================
// File: pe_exact.sv
// Description:
// Baseline Exact INT8 Multiply-Accumulate Processing Element (PE)
// with configurable pipelining (1 or 2 stages) and systolic forwarding.
// ============================================================================

`timescale 1ns / 1ps

module pe_exact #(
    parameter integer PIPELINE_STAGES = 2
)(
    input  logic                    clk,
    input  logic                    rst_n,

    // Control
    input  logic                    pe_en,
    input  logic                    clr_acc,
    input  logic                    use_ext_acc,

    // Data inputs
    input  logic signed [7:0]       act_in,
    input  logic signed [7:0]       weight_in,
    input  logic signed [31:0]      acc_in,

    // Systolic outputs
    output logic signed [7:0]       act_out,
    output logic signed [7:0]       weight_out,
    output logic signed [31:0]      acc_out
);

    // =========================================================================
    // Parameter check
    // =========================================================================

    initial begin
        if ((PIPELINE_STAGES != 1) && (PIPELINE_STAGES != 2)) begin
            $error("PIPELINE_STAGES must be 1 or 2. Got %0d",
                   PIPELINE_STAGES);
        end
    end

    // =========================================================================
    // 1-Stage Implementation
    // =========================================================================

    if (PIPELINE_STAGES == 1) begin : gen_pipe_1

        logic signed [15:0] mul_comb;
        logic signed [31:0] prod_sign_ext;
        logic signed [31:0] base_val;
        logic signed [31:0] next_acc;

        // Signed INT8 x INT8 multiplication
        assign mul_comb = act_in * weight_in;

        // Sign extend 16-bit product to 32 bits
        assign prod_sign_ext = {{16{mul_comb[15]}}, mul_comb};

        // Select accumulator source
        assign base_val = use_ext_acc ? acc_in : acc_out;

        // MAC result
        assign next_acc = base_val + prod_sign_ext;

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                act_out    <= 8'sd0;
                weight_out <= 8'sd0;
                acc_out    <= 32'sd0;
            end
            else if (clr_acc) begin
                acc_out <= 32'sd0;
            end
            else if (pe_en) begin
                act_out    <= act_in;
                weight_out <= weight_in;
                acc_out    <= next_acc;
            end
        end

    end

    // =========================================================================
    // 2-Stage Implementation
    // =========================================================================

    else begin : gen_pipe_2

        // ---------------------------------------------------------------------
        // Stage 1
        // ---------------------------------------------------------------------

        logic signed [15:0] mul_comb;
        logic signed [15:0] r_mul_out;
        logic signed [31:0] r_acc_in;

        // Combinational multiplier
        assign mul_comb = act_in * weight_in;

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                act_out    <= 8'sd0;
                weight_out <= 8'sd0;
                r_mul_out  <= 16'sd0;
                r_acc_in   <= 32'sd0;
            end
            else if (clr_acc) begin
                r_mul_out <= 16'sd0;
                r_acc_in  <= 32'sd0;
            end
            else if (pe_en) begin
                // Forward input operands
                act_out    <= act_in;
                weight_out <= weight_in;

                // Pipeline multiply result and accumulator input
                r_mul_out <= mul_comb;
                r_acc_in  <= acc_in;
            end
        end

        // ---------------------------------------------------------------------
        // Stage 2
        // ---------------------------------------------------------------------

        logic signed [31:0] prod_sign_ext;
        logic signed [31:0] base_val;
        logic signed [31:0] next_acc;

        // Sign extend multiplication result
        assign prod_sign_ext = {{16{r_mul_out[15]}}, r_mul_out};

        // Select external or internal accumulator
        assign base_val = use_ext_acc ? r_acc_in : acc_out;

        // Final MAC result
        assign next_acc = base_val + prod_sign_ext;

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                acc_out <= 32'sd0;
            end
            else if (clr_acc) begin
                acc_out <= 32'sd0;
            end
            else if (pe_en) begin
                acc_out <= next_acc;
            end
        end

    end

endmodule