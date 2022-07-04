from typing import Tuple

import torch


def generate_square_subsequent_mask(size: int, size_with_padding: int, device: torch.device) -> torch.Tensor:
    """

    Generate a square mask where values after the current position are masked out.
    Masked out positions have the value -10000.0, non masked position the value 0.
    The mask is supposed to be added on top of values before a softmax operation.

    Example with size=3 and size_with_padding=3:
        [[0, -10000.0, -10000.0],
         [0,        0, -10000.0],
         [0,        0,        0]]

    Example with size=2 and size_with_padding=3:
        [[       0, -10000.0, -10000.0],
         [       0,        0, -10000.0],
         [-10000.0, -10000.0, -10000.0]]

    :param size: Size of the mask
    :param size_with_padding: Size of the padded mask
    :param device: Pytorch device
    :return: tensor of shape [S, S]
    """
    assert size_with_padding >= size, f"'size_with_padding' must be greater or equal to size, " \
                                      f"but got {size_with_padding} and {size}"
    padded_mask: torch.Tensor = torch.triu(
        torch.full(size=(size_with_padding, size_with_padding),
                   fill_value=float("-inf"),
                   dtype=torch.float,
                   device=device),
        diagonal=1
    )
    padded_mask[size:, :] = float("-inf")
    padded_mask[:, size:] = float("-inf")
    return padded_mask


def extend_tgt_mask(tgt_mask: torch.Tensor) -> torch.Tensor:
    """

    Extends a batch of tensors indicating padded positions to a batch of square subsequent masks.

    :param tgt_mask: tensor of shape [B, S]
    :return: tensor of shape [B, S, S]
    """
    assert tgt_mask.dim() == 2, f"expected mask to have two dimensions, but got a shape of {tgt_mask.shape}"
    B, S = tgt_mask.size()
    sizes, size_with_padding = get_sizes_and_size_with_padding(tgt_mask)
    extended_mask = torch.zeros((B, S, S), dtype=torch.float, device=tgt_mask.device)
    for i in range(B):
        extended_mask[i] = generate_square_subsequent_mask(sizes[i], size_with_padding, device=tgt_mask.device)
    return extended_mask


def generate_square_mask(size: int, size_with_padding: int, device: torch.device) -> torch.Tensor:
    """

    Generate a square mask.
    Masked out positions have the value -10000.0, non masked position the value 0.
    The mask is supposed to be added on top of values before a softmax operation.

    Example with size=3 and size_with_padding=3:
        [[0,        0,        0],
         [0,        0,        0],
         [0,        0,        0]]

    Example with size=2 and size_with_padding=3:
        [[       0,        0, -10000.0],
         [       0,        0, -10000.0],
         [-10000.0, -10000.0, -10000.0]]

    :param size: Size of the mask
    :param size_with_padding: Size of the padded mask
    :param device: Pytorch device
    :return: tensor of shape [S, S]
    """
    assert size_with_padding >= size, "'size_with_padding' must be greater or equal to size"
    padded_mask = torch.full(size=(size_with_padding, size_with_padding),
                             fill_value=float("-inf"),
                             dtype=torch.float,
                             device=device)
    padded_mask[:size, :size] = 0.0
    return padded_mask


def extend_src_mask(src_mask: torch.Tensor) -> torch.Tensor:
    """

    Extends a batch of tensors indicating padded positions to a batch of square masks.

    :param src_mask: tensor of shape [B, S]
    :return: tensor of shape [B, S, S]
    """
    assert src_mask.dim() == 2, f"expected mask to have two dimensions, but got a shape of {src_mask.shape}"
    B, S = src_mask.size()
    sizes, size_with_padding = get_sizes_and_size_with_padding(src_mask)
    extended_mask = torch.zeros((B, S, S), dtype=torch.float, device=src_mask.device)
    for i in range(B):
        extended_mask[i] = generate_square_mask(sizes[i], size_with_padding, device=src_mask.device)
    return extended_mask


def generate_rectangular_mask(size: Tuple[int, int],
                              size_with_padding: Tuple[int, int],
                              device: torch.device) -> torch.Tensor:
    """

    Generate a rectangular mask.
    Masked out positions have the value -10000.0, non masked position the value 0.
    The mask is supposed to be added on top of values before a softmax operation.

    Example with size=(3, 2) and size_with_padding=(3, 2):
        [[0,        0],
         [0,        0],
         [0,        0]]

    Example with size=(3, 2) and size_with_padding=(3, 3):
        [[       0,        0, -10000.0],
         [       0,        0, -10000.0],
         [       0,        0, -10000.0]]

    :param size: Tuple containing number of rows and columns of the mask
    :param size_with_padding: Tuple containing number of rows and columns of the padded mask
    :param device: Pytorch device
    :return: tensor of shape [T, S]
    """
    assert size_with_padding[0] >= size[0] \
           and size_with_padding[1] >= size[1], "both elements in 'size_with_padding' must be greater or equal to " \
                                                "the respective elements in 'size'"
    padded_mask = torch.full(size=size_with_padding,
                             fill_value=float("-inf"),
                             dtype=torch.float,
                             device=device)
    padded_mask[:size[0], :size[1]] = 0.0
    return padded_mask


def extend_memory_mask(memory_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
    """

   Extends a batch of tensors indicating padded positions to a batch of rectangular masks.

   :param memory_mask: tensor of shape [B, S]
   :param tgt_mask: tensor of shape [B, T, T]
   :return: tensor of shape [B, T, S]
   """
    assert memory_mask.dim() == 2, \
        f"expected memory_mask to have two dimensions, but got a shape of {memory_mask.shape}"
    assert tgt_mask.dim() == 3, \
        f"expected tgt_mask to have three dimensions (already extended), but got a shape of {tgt_mask.shape}"
    B, S = memory_mask.size()
    _, T, _ = tgt_mask.size()
    sizes, size_with_padding = get_sizes_and_size_with_padding(memory_mask)
    extended_mask = torch.zeros((B, T, S), dtype=torch.float, device=memory_mask.device)
    for i in range(B):
        extended_mask[i] = generate_rectangular_mask((T, sizes[i]),
                                                     (T, size_with_padding),
                                                     device=memory_mask.device)
    return extended_mask


def get_sizes_and_size_with_padding(mask: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """

    From a batch of tensors indicating padded positions (where the tensor is equal to 0)
    determine the sizes and sizes with padding to be used to generate masks.

    Example with tensor [[1, 1, 1, 0], [1, 1, 0, 0]]:
        -> ([3, 2], 4)
        The first tensor has an unpadded length of 3, the second tensor has an unpadded length of 2.
        Both tensors are padded to length 4.

    :param mask: tensor of shape [B, S]
    :return: tensor indicating sizes of the tensors in the batch, padded size
    """
    assert mask.dim() == 2, f"expected mask to have two dimensions, but got a shape of {mask.shape}"
    B, S = mask.size()
    return mask.sum(dim=1).long(), S


def get_attention_masks(attention_mask: torch.Tensor, target_attention_mask: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """

    Convenience function to generate src_mask, tgt_mask and memory_mask
    from attention_masks indicating padded positions.

    :param attention_mask: tensor of shape [B, S]
    :param target_attention_mask: tensor of shape [B, T]
    :return: tensors of shape [B, S, S], [B, T, T], [B, T, S]
    """
    # S and T should be the same (because the sequences are padded to the same length)
    B, S = attention_mask.size()
    B, T = target_attention_mask.size()

    src_mask = torch.zeros((B, S, S), dtype=torch.float, device=attention_mask.device)
    tgt_mask = torch.zeros((B, T, T), dtype=torch.float, device=target_attention_mask.device)
    memory_mask = torch.zeros((B, T, S), dtype=torch.float, device=attention_mask.device)

    src_sizes, src_size_with_padding = get_sizes_and_size_with_padding(attention_mask)
    tgt_sizes, tgt_size_with_padding = get_sizes_and_size_with_padding(target_attention_mask)

    for i in range(B):
        src_mask[i] = generate_square_mask(size=src_sizes[i],
                                           size_with_padding=src_size_with_padding,
                                           device=attention_mask.device)
        tgt_mask[i] = generate_square_subsequent_mask(size=tgt_sizes[i],
                                                      size_with_padding=tgt_size_with_padding,
                                                      device=target_attention_mask.device)
        memory_mask[i] = generate_rectangular_mask(size=(tgt_sizes[i], src_sizes[i]),
                                                   size_with_padding=(tgt_size_with_padding, src_size_with_padding),
                                                   device=attention_mask.device)

    return src_mask, tgt_mask, memory_mask


def get_attention_masks_from_ids(input_ids: torch.Tensor, target_input_ids: torch.Tensor, pad_token_id: int) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """

    Convenience function to generate src_mask, tgt_mask, and memory_mask
    from input_ids.

    :param input_ids: tensor of shape [B, S]
    :param target_input_ids: tensor of shape [B, T]
    :param pad_token_id: id of the padding token
    :return: tensors of shape [B, S, S], [B, T, T], [B, T, S]
    """
    # noinspection PyTypeChecker
    attention_mask: torch.Tensor = (input_ids != pad_token_id)
    # noinspection PyTypeChecker
    target_attention_mask: torch.Tensor = (target_input_ids != pad_token_id)
    return get_attention_masks(attention_mask, target_attention_mask)


def get_padding_mask(input_ids: torch.Tensor, padding_token_id: int) -> torch.Tensor:
    """

    :param input_ids: tensor of shape [L, B]
    :param padding_token_id: id of the padding token
    """
    return (torch.transpose(input_ids, 0, 1) == padding_token_id).to(torch.bool)
