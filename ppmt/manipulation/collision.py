"""Utilities for extracting collision information."""

from typing import Any, Tuple
import numpy as np
import mujoco


def get_collision_info(
    contact: Any, geom1: int, geom2: int
):
  dist = 0.0
  normal=[0.,0.,0.]
  if len(contact.geom) != 0:
    mask = (np.array([geom1, geom2]) == contact.geom).all(axis=1)
    mask |= (np.array([geom2, geom1]) == contact.geom).all(axis=1)
    idx = np.where(mask, contact.dist, 1e4).argmin()

    dist = contact.dist[idx] * mask[idx]
    normal = (dist < 0) * contact.frame[idx, :3]
  
  return dist,normal

def geoms_colliding(data:mujoco.MjData,geom1:int,geom2:int):
  """Return True if the two geoms are colliding."""
  return get_collision_info(data.contact, geom1, geom2)[0] < 0
