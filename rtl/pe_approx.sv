`timescale 1ns / 1ps

module pe_approx #(
    parameter integer PIPELINE_STAGES = 2,
    parameter integer TRUNC_BITS      = 4,
    parameter bit     ROUND_BIAS      = 1
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
            $error("PIPELINE_STAGES must be 1 or 2. Got %0d", PIPELINE_STAGES);
        end
    end

    // Combinational approximate multiplier instance
    logic signed [15:0] mul_approx_comb;

    approx_mult #(
        .WIDTH      (8),
        .TRUNC_BITS (TRUNC_BITS),
        .ROUND_BIAS (ROUND_BIAS)
    ) u_approx_mult (
        .a          (act_in),
        .b          (weight_in),
        .product    (mul_approx_comb)
    );

    // =========================================================================
    // 1-Stage Implementation
    // =========================================================================
    if (PIPELINE_STAGES == 1) begin : gen_pipe_1

        logic signed [31:0] prod_sign_ext;
        logic signed [31:0] base_val;
        logic signed [31:0] next_acc;

        assign prod_sign_ext = {{16{mul_approx_comb[15]}}, mul_approx_comb};
        assign base_val      = use_ext_acc ? acc_in : acc_out;
        assign next_acc      = base_val + prod_sign_ext;

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                act_out    <= 8'sd0;
                weight_out <= 8'sd0;
                acc_out    <= 32'sd0;
            end
            else if (clr_acc) begin
                acc_out    <= 32'sd0;
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

        // Stage 1 Registers
        logic signed [15:0] r_mul_out;
        logic signed [31:0] r_acc_in;

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                act_out    <= 8'sd0;
                weight_out <= 8'sd0;
                r_mul_out  <= 16'sd0;
                r_acc_in   <= 32'sd0;
            end
            else if (clr_acc) begin
                r_mul_out  <= 16'sd0;
                r_acc_in   <= 32'sd0;
            end
            else if (pe_en) begin
                act_out    <= act_in;
                weight_out <= weight_in;
                r_mul_out  <= mul_approx_comb;
                r_acc_in   <= acc_in;
            end
        end

        // Stage 2 Accumulation
        logic signed [31:0] prod_sign_ext;
        logic signed [31:0] base_val;
        logic signed [31:0] next_acc;

        assign prod_sign_ext = {{16{r_mul_out[15]}}, r_mul_out};
        assign base_val      = use_ext_acc ? r_acc_in : acc_out;
        assign next_acc      = base_val + prod_sign_ext;

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