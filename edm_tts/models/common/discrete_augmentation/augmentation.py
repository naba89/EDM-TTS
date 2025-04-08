import torch
import torch.nn.functional as F


def time_warp(input_tensor, attn_mask, W=80):
    """
    Applies time warping to each sequence in the batch individually, taking into account variable lengths.

    Args:
    input_tensor (torch.Tensor): Tensor of shape (B, T) or (B, T, D), where B is batch size, T is time steps,
                                 and D is feature dimension (optional).
    attn_mask (torch.Tensor): Attention mask of shape (B, T), where 1 indicates valid tokens and 0 indicates padding.
    W (int): Time warp factor (default: 80).

    Returns:
    torch.Tensor: Warped tensor of the same shape as input_tensor, with padding regions left unchanged.
    """
    B, T = input_tensor.shape[:2]  # Get batch size and time steps
    D = input_tensor.shape[2] if input_tensor.dim() == 3 else 1  # Handle both (B, T, D) and (B, T) cases
    orig_dtype = input_tensor.dtype  # Save original data type for later

    input_tensor = input_tensor.float()  # Convert input to float for interpolation

    # Reshape (B, T) input to (B, T, 1) for consistent processing
    if D == 1:
        input_tensor = input_tensor.unsqueeze(-1)

    warped_output = torch.zeros_like(input_tensor)  # Output tensor initialized to zeros (for padding areas)

    for i in range(B):
        # Get valid sequence length based on attention mask
        valid_length = attn_mask[i].sum().long()

        # Skip warping if the valid sequence length is too small
        if valid_length <= 2 * W:
            warped_output[i, :valid_length, :] = input_tensor[i, :valid_length, :]  # Just copy the sequence
            continue

        # Select a random center C within the valid sequence
        C = torch.randint(W + 1, valid_length - W, (1,)).item()

        # Determine warped size S
        S = torch.randint(C - W, C + W + 1, (1,)).item()

        # Separate valid input into left and right parts based on C
        left_part = input_tensor[i, :C, :]  # Shape: (C, D)
        right_part = input_tensor[i, C:valid_length, :]  # Shape: (valid_length - C, D)

        # Warp the left and right parts
        left_warped = F.interpolate(left_part.unsqueeze(0).permute(0, 2, 1),
                                    size=S, mode='nearest').squeeze(0).permute(1, 0)  # Shape: (S, D)
        right_warped = F.interpolate(right_part.unsqueeze(0).permute(0, 2, 1),
                                     size=valid_length - S,
                                     mode='nearest').squeeze(0).permute(1, 0)  # Shape: (valid_length - S, D)

        # Combine the warped parts and place them in the output tensor
        warped_output[i, :valid_length, :] = torch.cat([left_warped, right_warped], dim=0)

    # If the input was originally of shape (B, T), return (B, T)
    if D == 1:
        warped_output = warped_output.squeeze(-1)

    return warped_output.to(orig_dtype)  # Convert back to original data type
