# Room Map Semantic Values (normalized to [-1, 1] range)
OBSTACLE = -1.0      # Known obstacles (discovered by collision or local view)
CLEAN = -0.1         # Clean/visited areas
ROBOT = 0.0          # Current robot location
UNKNOWN = 0.2        # Unexplored areas
DIRTY = 0.5          # Known dirty cells (from initial dirt_map)
RETURN_TARGET = 1.0  # Starting position after all dirt cleaned 
