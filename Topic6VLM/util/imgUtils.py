import cv2


def scale_to_max_dimension(
    image: cv2.typing.MatLike, max_dim_size: int
) -> cv2.typing.MatLike:
    """
    Scales an image to a maximum width or height, preserving aspect ratio.

    Args:
        image: The input image (NumPy array).
        max_dim_size: The maximum allowed size for either width or height.

    Returns:
        The resized image.
    """
    height, width = image.shape[:2]

    # Check if the image already fits within the max dimension
    if max(height, width) <= max_dim_size:
        return image

    # Calculate the scaling ratio
    ratio = max_dim_size / max(width, height)

    # Calculate new dimensions
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    dim = (new_width, new_height)

    # Resize the image using the appropriate interpolation method
    # INTER_AREA is generally recommended for shrinking images
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    return resized_image
