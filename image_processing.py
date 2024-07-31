import numpy as np

def windowing(
    img: np.ndarray, wl: int = 0, ww: int = 400, mode: str = "uint8"
) -> np.ndarray:
    """windowing process.

    Args:
        img (numpy.ndarray): The input image to filter.
        wl (int): The center of the window.
        ww (int): The width of the window.
        mode (str): The output data type of the filtered image.
            One of {'uint8', 'uint16', 'float32'}.

    Returns:
        numpy.ndarray: The filtered image.

    Raises:
        ValueError: If the mode is not one of {'uint8', 'uint16', 'float32'}.
    """
    floor, ceil = wl - ww // 2, wl + ww // 2
    img = np.clip(img, floor, ceil)
    if mode == "uint8":
        img = (((img - floor) / (ceil - floor)) * 255).astype(np.uint8)
    elif mode == "uint16":
        img = (((img - floor) / (ceil - floor)) * 65535).astype(np.uint16)
    elif mode == "float32":
        img = ((img - floor) / (ceil - floor)).astype(np.float32)
    else:
        raise Exception(f"unexpected mode: {mode}")

    return img

def mri_normalization(img: np.ndarray) -> np.ndarray:
    """MRI画像の正規化.
    Args:
        img (numpy.ndarray): MRI信号の2次元の入力画像.
    Returns:
        numpy.ndarray: 0~1に正規化された画像.
    Note:
        外れ値の除去を含むので、元の信号を復元することはできない.
    """
    img = img.astype(np.float32) / np.percentile(img, 99)
    return img