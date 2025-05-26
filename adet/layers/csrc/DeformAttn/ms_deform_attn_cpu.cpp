/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#include "ms_deform_attn_cpu.h"
#include <torch/extension.h>
#include <vector>
#include <math.h>

template <typename scalar_t>
scalar_t ms_deform_attn_im2col_bilinear_cpu(const scalar_t* bottom_data,
                                            const int height,
                                            const int width,
                                            const int n_heads,
                                            const int channels,
                                            scalar_t h,
                                            scalar_t w,
                                            const int m,
                                            const int c) {
    const int h_low = floor(h);
    const int w_low = floor(w);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;

    const scalar_t lh = h - h_low;
    const scalar_t lw = w - w_low;
    const scalar_t hh = 1 - lh, hw = 1 - lw;

    const int w_stride = n_heads * channels;
    const int h_stride = width * w_stride;
    const int h_low_ptr_offset = h_low * h_stride;
    const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
    const int w_low_ptr_offset = w_low * w_stride;
    const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
    const int base_ptr = m * channels + c;

    scalar_t v1 = 0, v2 = 0, v3 = 0, v4 = 0;
    if (h_low >= 0 && w_low >= 0) {
        int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
        v1 = bottom_data[ptr1];
    }
    if (h_low >= 0 && w_high <= width - 1) {
        int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
        v2 = bottom_data[ptr2];
    }
    if (h_high <= height - 1 && w_low >= 0) {
        int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
        v3 = bottom_data[ptr3];
    }
    if (h_high <= height - 1 && w_high <= width - 1) {
        int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
        v4 = bottom_data[ptr4];
    }

    const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

template <typename scalar_t>
void ms_deform_attn_col2im_bilinear_cpu(const scalar_t* bottom_data,
                                        const int height,
                                        const int width,
                                        const int n_heads,
                                        const int channels,
                                        scalar_t h,
                                        scalar_t w,
                                        const int m,
                                        const int c,
                                        const scalar_t top_grad,
                                        const scalar_t attn_weight,
                                        scalar_t* grad_value,
                                        scalar_t& grad_h,
                                        scalar_t& grad_w,
                                        scalar_t& grad_attn_weight) {
    const int h_low = floor(h);
    const int w_low = floor(w);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;

    const scalar_t lh = h - h_low;
    const scalar_t lw = w - w_low;
    const scalar_t hh = 1 - lh, hw = 1 - lw;

    const int w_stride = n_heads * channels;
    const int h_stride = width * w_stride;
    const int h_low_ptr_offset = h_low * h_stride;
    const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
    const int w_low_ptr_offset = w_low * w_stride;
    const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
    const int base_ptr = m * channels + c;

    scalar_t v1 = 0, v2 = 0, v3 = 0, v4 = 0;
    scalar_t grad_h_weight = 0, grad_w_weight = 0;
    scalar_t grad_val = 0;

    if (h_low >= 0 && w_low >= 0) {
        const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
        v1 = bottom_data[ptr1];
        grad_h_weight -= hw * v1;
        grad_w_weight -= hh * v1;
        grad_value[ptr1] += hh * hw * attn_weight * top_grad;
    }
    if (h_low >= 0 && w_high <= width - 1) {
        const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
        v2 = bottom_data[ptr2];
        grad_h_weight -= lw * v2;
        grad_w_weight += hh * v2;
        grad_value[ptr2] += hh * lw * attn_weight * top_grad;
    }
    if (h_high <= height - 1 && w_low >= 0) {
        const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
        v3 = bottom_data[ptr3];
        grad_h_weight += hw * v3;
        grad_w_weight -= lh * v3;
        grad_value[ptr3] += lh * hw * attn_weight * top_grad;
    }
    if (h_high <= height - 1 && w_high <= width - 1) {
        const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
        v4 = bottom_data[ptr4];
        grad_h_weight += lw * v4;
        grad_w_weight += lh * v4;
        grad_value[ptr4] += lh * lw * attn_weight * top_grad;
    }

    const scalar_t val = (hh * hw * v1 + hh * lw * v2 + lh * hw * v3 + lh * lw * v4);
    grad_attn_weight = top_grad * val;
    grad_h = grad_h_weight * attn_weight * top_grad * height;
    grad_w = grad_w_weight * attn_weight * top_grad * width;
}

at::Tensor ms_deform_attn_cpu_forward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step) {
    AT_ASSERTM(value.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(spatial_shapes.is_contiguous(), "spatial_shapes tensor has to be contiguous");
    AT_ASSERTM(level_start_index.is_contiguous(), "level_start_index tensor has to be contiguous");
    AT_ASSERTM(sampling_loc.is_contiguous(), "sampling_loc tensor has to be contiguous");
    AT_ASSERTM(attn_weight.is_contiguous(), "attn_weight tensor has to be contiguous");

    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    const int channels = value.size(3);
    const int num_levels = spatial_shapes.size(0);
    const int num_query = sampling_loc.size(1);
    const int num_point = sampling_loc.size(4);

    const int im2col_step_ = std::min(batch, im2col_step);
    AT_ASSERTM(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);

    auto output = at::zeros({batch, num_query, num_heads, channels}, value.options());
    const int batch_n = im2col_step_;
    auto output_n = output.view({batch/im2col_step_, batch_n, num_query, num_heads, channels});

    const int per_value_size = spatial_size * num_heads * channels;
    const int per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
    const int per_attn_weight_size = num_query * num_heads * num_levels * num_point;

    auto spatial_shapes_data = spatial_shapes.data_ptr<int64_t>();
    auto level_start_index_data = level_start_index.data_ptr<int64_t>();

    for (int n = 0; n < batch / im2col_step_; ++n) {
        auto columns = output_n.select(0, n);
        AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_forward_cpu", ([&] {
            auto value_data = value.data_ptr<scalar_t>();
            auto sampling_loc_data = sampling_loc.data_ptr<scalar_t>();
            auto attn_weight_data = attn_weight.data_ptr<scalar_t>();
            auto columns_data = columns.data_ptr<scalar_t>();

            for (int b = 0; b < batch_n; ++b) {
                for (int q = 0; q < num_query; ++q) {
                    for (int h = 0; h < num_heads; ++h) {
                        for (int c = 0; c < channels; ++c) {
                            scalar_t val = 0;
                            for (int l = 0; l < num_levels; ++l) {
                                const int level_start_id = level_start_index_data[l];
                                const int spatial_h = spatial_shapes_data[l * 2];
                                const int spatial_w = spatial_shapes_data[l * 2 + 1];
                                const int loc_offset = ((n * im2col_step_ + b) * num_query + q) * num_heads * num_levels * num_point * 2 + h * num_levels * num_point * 2 + l * num_point * 2;
                                const int weight_offset = ((n * im2col_step_ + b) * num_query + q) * num_heads * num_levels * num_point + h * num_levels * num_point + l * num_point;

                                for (int p = 0; p < num_point; ++p) {
                                    const scalar_t loc_w = sampling_loc_data[loc_offset + p * 2];
                                    const scalar_t loc_h = sampling_loc_data[loc_offset + p * 2 + 1];
                                    const scalar_t weight = attn_weight_data[weight_offset + p];

                                    const scalar_t h_im = loc_h * spatial_h - 0.5;
                                    const scalar_t w_im = loc_w * spatial_w - 0.5;

                                    if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
                                        const scalar_t* value_ptr = value_data + (n * im2col_step_ + b) * per_value_size + level_start_id * num_heads * channels;
                                        val += ms_deform_attn_im2col_bilinear_cpu(value_ptr, spatial_h, spatial_w, num_heads, channels, h_im, w_im, h, c) * weight;
                                    }
                                }
                            }
                            columns_data[((b * num_query + q) * num_heads + h) * channels + c] = val;
                        }
                    }
                }
            }
        }));
    }

    output = output.view({batch, num_query, num_heads * channels});
    return output;
}

std::vector<at::Tensor> ms_deform_attn_cpu_backward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int im2col_step) {
    AT_ASSERTM(value.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(spatial_shapes.is_contiguous(), "spatial_shapes tensor has to be contiguous");
    AT_ASSERTM(level_start_index.is_contiguous(), "level_start_index tensor has to be contiguous");
    AT_ASSERTM(sampling_loc.is_contiguous(), "sampling_loc tensor has to be contiguous");
    AT_ASSERTM(attn_weight.is_contiguous(), "attn_weight tensor has to be contiguous");
    AT_ASSERTM(grad_output.is_contiguous(), "grad_output tensor has to be contiguous");

    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    const int channels = value.size(3);
    const int num_levels = spatial_shapes.size(0);
    const int num_query = sampling_loc.size(1);
    const int num_point = sampling_loc.size(4);

    const int im2col_step_ = std::min(batch, im2col_step);
    AT_ASSERTM(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);

    auto grad_value = at::zeros_like(value);
    auto grad_sampling_loc = at::zeros_like(sampling_loc);
    auto grad_attn_weight = at::zeros_like(attn_weight);

    const int per_value_size = spatial_size * num_heads * channels;
    const int per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
    const int per_attn_weight_size = num_query * num_heads * num_levels * num_point;

    auto spatial_shapes_data = spatial_shapes.data_ptr<int64_t>();
    auto level_start_index_data = level_start_index.data_ptr<int64_t>();

    for (int n = 0; n < batch / im2col_step_; ++n) {
        auto grad_output_n = grad_output.view({batch/im2col_step_, im2col_step_, num_query, num_heads, channels}).select(0, n);
        AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cpu", ([&] {
            auto value_data = value.data_ptr<scalar_t>();
            auto sampling_loc_data = sampling_loc.data_ptr<scalar_t>();
            auto attn_weight_data = attn_weight.data_ptr<scalar_t>();
            auto grad_output_data = grad_output_n.data_ptr<scalar_t>();
            auto grad_value_data = grad_value.data_ptr<scalar_t>();
            auto grad_sampling_loc_data = grad_sampling_loc.data_ptr<scalar_t>();
            auto grad_attn_weight_data = grad_attn_weight.data_ptr<scalar_t>();

            for (int b = 0; b < im2col_step_; ++b) {
                for (int q = 0; q < num_query; ++q) {
                    for (int h = 0; h < num_heads; ++h) {
                        for (int c = 0; c < channels; ++c) {
                            const scalar_t top_grad = grad_output_data[((b * num_query + q) * num_heads + h) * channels + c];
                            for (int l = 0; l < num_levels; ++l) {
                                const int level_start_id = level_start_index_data[l];
                                const int spatial_h = spatial_shapes_data[l * 2];
                                const int spatial_w = spatial_shapes_data[l * 2 + 1];
                                const int loc_offset = ((n * im2col_step_ + b) * num_query + q) * num_heads * num_levels * num_point * 2 + h * num_levels * num_point * 2 + l * num_point * 2;
                                const int weight_offset = ((n * im2col_step_ + b) * num_query + q) * num_heads * num_levels * num_point + h * num_levels * num_point + l * num_point;

                                for (int p = 0; p < num_point; ++p) {
                                    const scalar_t loc_w = sampling_loc_data[loc_offset + p * 2];
                                    const scalar_t loc_h = sampling_loc_data[loc_offset + p * 2 + 1];
                                    const scalar_t weight = attn_weight_data[weight_offset + p];

                                    const scalar_t h_im = loc_h * spatial_h - 0.5;
                                    const scalar_t w_im = loc_w * spatial_w - 0.5;

                                    scalar_t grad_h = 0, grad_w = 0, grad_attn = 0;
                                    if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
                                        const scalar_t* value_ptr = value_data + (n * im2col_step_ + b) * per_value_size + level_start_id * num_heads * channels;
                                        scalar_t* grad_value_ptr = grad_value_data + (n * im2col_step_ + b) * per_value_size + level_start_id * num_heads * channels;
                                        ms_deform_attn_col2im_bilinear_cpu(
                                            value_ptr, spatial_h, spatial_w, num_heads, channels,
                                            h_im, w_im, h, c, top_grad, weight,
                                            grad_value_ptr, grad_h, grad_w, grad_attn);
                                    }
                                    grad_sampling_loc_data[loc_offset + p * 2] += grad_w;
                                    grad_sampling_loc_data[loc_offset + p * 2 + 1] += grad_h;
                                    grad_attn_weight_data[weight_offset + p] += grad_attn;
                                }
                            }
                        }
                    }
                }
            }
        }));
    }

    return {grad_value, grad_sampling_loc, grad_attn_weight};
}
