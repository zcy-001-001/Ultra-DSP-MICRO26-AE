"""
INT8 GEMV shape definitions for the CPU memory-bound benchmark.
"""

GEMV_CONFIGS = [
    ("GEMV_1024x1024", 1, 1024, 1024),
    ("GEMV_2048x2048", 1, 2048, 2048),
    ("GEMV_4096x4096", 1, 4096, 4096),
    ("GEMV_4096x12288", 1, 4096, 12288),
    ("GEMV_4096x16384", 1, 4096, 16384),
    ("GEMV_8192x8192", 1, 8192, 8192),
]
