import cv2
from typing import Tuple

# # Aruco marker dictionary
# ARUCO_DICT: int = cv2.aruco.DICT_4X4_250

# # Number of squares in the vertical and horizontal directions on the calibration board
# SQUARES_VERTICALLY: int = 16
# SQUARES_HORIZONTALLY: int = 31

# # Length of each square on the calibration board and length of the markers (in metres)
# SQUARE_LENGTH: float = 0.04933
# MARKER_LENGTH: float = 0.03846

# # Size of the calibration board image
# BOARD_IMAGE_SIZE: Tuple[int, int] = (2100, 1100)

# # Size of the margin in pixels
# MARGIN_PX: int = 0

# ArUco marker dictionary
ARUCO_DICT: int = cv2.aruco.DICT_5X5_100  # or DICT_5X5_250 if you know it’s the larger set

# Number of squares in the vertical and horizontal directions on the calibration board
SQUARES_VERTICALLY: int = 9
SQUARES_HORIZONTALLY: int = 14

# Length of each square on the calibration board and length of the markers (in metres)
SQUARE_LENGTH: float = 0.020   # 20 mm
MARKER_LENGTH: float = 0.016   # 16 mm

# Size of the calibration board image (only needed if generating/printing the board)
BOARD_IMAGE_SIZE: Tuple[int, int] = (2100, 1100)  # can leave as is, not relevant if you already have a board

# Size of the margin in pixels
MARGIN_PX: int = 0

