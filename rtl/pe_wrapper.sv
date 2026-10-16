`timescale 1ns / 1ps

module pe_wrapper #(
    parameter integer PIPELINE_STAGES = 2,
    parameter bit     USE_APPROX      = 0, // 0: Exact PE, 1: Approx PE
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

    // Macro-based override logic
    localparam bit SELECT_APPROX =
`ifdef APPROX_PE
        1'b1;
`else
        USE_APPROX;
`endif

    generate
        if (SELECT_APPROX) begin : gen_approx_pe
            pe_approx #(
                .PIPELINE_STAGES (PIPELINE_STAGES),
                .TRUNC_BITS      (TRUNC_BITS),
                .ROUND_BIAS      (ROUND_BIAS)
            ) u_pe (
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
        end
        else begin : gen_exact_pe
            pe_exact #(
                .PIPELINE_STAGES (PIPELINE_STAGES)
            ) u_pe (
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
        end
    endgenerate

endmodule